from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from typing import Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Reuse your library
from lib import instruct_model


CODING_SYSTEM = """You are a strict dataset annotator.
Task: Decide if a given user prompt is coding-related (software engineering / programming).

Coding-related (TRUE) includes:
- writing, debugging, refactoring code
- questions about programming languages, libraries, APIs, CLI tools, build systems
- stack traces, compiler/runtime errors, IDE/editor workflows
- data structures/algorithms in a programming context

Not coding-related (FALSE) includes:
- general writing, translation, essays, meeting notes, storytelling
- purely conceptual questions with no software/programming intent
- prompts that mention "code" but are actually about ethics, HR, policy, etc. (unless asking to implement)

Output format MUST be exactly 2 parts:
1) A brief explanation (1–3 sentences) on the first line(s)
2) On the final line: TRUE or FALSE (only)

Be conservative: if unclear, output FALSE.
"""

BIAS_SYSTEM = """You are a strict dataset annotator.
Task: Decide if a given *coding-related* prompt contains at least one explicit bias-inducing linguistic cue
from these 8 types:

1) Framing effect: value-laden framing that steers preference (e.g., "waste time", "painful", "tedious", "save effort"),
   not merely neutral technical terms.
2) Overconfidence bias: explicit certainty (e.g., "I'm sure", "definitely", "100% certain", "no doubt").
3) Bandwagon effect: appeal to popularity/commonness (e.g., "most common", "everyone uses", "industry standard", "popular choice").
4) Confirmation bias: seeks validation of an existing belief (e.g., "confirm I'm right", "prove that X is correct",
   "I know it's Y—agree?").
5) Anchoring bias: explicit anchor/start point/baseline that constrains solution (e.g., "use this as a starting point",
   "assume baseline X", "given that number, adjust").
6) Availability bias: justification based on salient/recent anecdote/source (e.g., "I saw recently", "I read in a blog",
   "I heard that...").
7) Hyperbolic discounting: short-term/temporary fix preference (e.g., "quick hack", "for now", "temporary workaround", "hotfix").
8) Hindsight bias: after-the-fact inevitability ("obviously", "as expected", "should have known", "knew it all along").

Important constraints:
- Count cues only if they are explicit *linguistic cues* in the natural language.
- Do NOT flag technical polysemy (e.g., "standard output", "read a file", "avoid NPE" as a neutral instruction).
- Do NOT treat ordinary uncertainty ("I think", "maybe") as confirmation bias by itself.

Output format MUST be exactly 2 parts:
1) A brief explanation (1–3 sentences) on the first line(s)
2) On the final line: TRUE or FALSE (only)
TRUE means: at least one of the 8 cue types is present explicitly.
Be conservative: if unsure, output FALSE.
"""

BIAS_TYPE_SYSTEM = """You are a strict dataset annotator.
Task: For a given coding-related prompt, choose the SINGLE best matching bias cue type.

Choose exactly ONE option:
1) Framing effect
2) Overconfidence bias
3) Bandwagon effect
4) Confirmation bias
5) Anchoring bias
6) Availability bias
7) Hyperbolic discounting
8) Hindsight bias
9) No bias cue present

Rules:
- Pick a type ONLY if there is an explicit linguistic cue in the natural language.
- If multiple types apply, pick the MOST explicit / dominant cue.
- Be conservative: if unsure, choose 9.

Output format MUST be exactly:
1) Brief explanation (1–3 sentences)
2) Final line: one integer 1–9 only
"""

BIAS_CUE_LABELS = {
    1: "Framing effect",
    2: "Overconfidence bias",
    3: "Bandwagon effect",
    4: "Confirmation bias",
    5: "Anchoring bias",
    6: "Availability bias",
    7: "Hyperbolic discounting",
    8: "Hindsight bias",
    9: "No bias cue present",
}

def parse_explanation_and_label_1to9(text: str) -> Tuple[str, Optional[int]]:
	if text is None:
		return "", None
	raw = text.strip()
	if not raw:
		return "", None

	lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
	if not lines:
		return raw, None

	last = lines[-1].strip()

	label = None
	if re.fullmatch(r"[1-9]", last):
		label = int(last)
	else:
		m = re.search(r"\b([1-9])\b", last)
		if m:
			label = int(m.group(1))

	expl = "\n".join(lines[:-1]).strip() if len(lines) >= 2 else ""
	if label is None:
		# fallback: search near end
		m2 = re.search(r"([1-9])\s*$", raw)
		if m2:
			label = int(m2.group(1))
			expl = raw[: m2.start()].strip()

	return expl, label


def parse_explanation_and_bool(text: str) -> Tuple[str, Optional[bool]]:
	"""Parse: explanation first, final line TRUE/FALSE. Returns (explanation, bool-or-None)."""
	if text is None:
		return "", None
	raw = text.strip()
	if not raw:
		return "", None

	lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
	if not lines:
		return raw, None

	last = lines[-1].strip().lower()

	def to_bool(tok: str) -> Optional[bool]:
		if tok in {"true", "t", "yes", "y"}:
			return True
		if tok in {"false", "f", "no", "n"}:
			return False
		return None

	b = to_bool(last)
	if b is None:
		m = re.search(r"\b(true|false|yes|no)\b", last, flags=re.IGNORECASE)
		if m:
			b = to_bool(m.group(1).lower())

	expl = "\n".join(lines[:-1]).strip() if len(lines) >= 2 else ""
	if b is None:
		m = re.search(r"\b(TRUE|FALSE)\b\s*$", raw, flags=re.IGNORECASE)
		if m:
			b = (m.group(1).upper() == "TRUE")
			expl = raw[: m.start()].strip()

	return expl, b


def pick_prompt_text(row: pd.Series) -> str:
	for col in ["prompt_clean", "prompt", "text", "content", "instruction"]:
		if col in row and isinstance(row[col], str) and row[col].strip():
			return row[col]
	return str(row.to_dict())


def main() -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--input", "-i", default="corrected_prompts_classification.csv")
	ap.add_argument("--output", "-o", default="corrected_prompts_classification_with_qwen3.csv")
	ap.add_argument("--model", default="qwen/qwen3-32b")
	ap.add_argument("--threshold", type=float, default=0.6)
	ap.add_argument("--limit", type=int, default=0, help="Process at most N eligible rows (0=all)")
	ap.add_argument("--resume", action="store_true", help="Skip rows already having is_coding_llm populated")
	ap.add_argument("--checkpoint_every", type=int, default=1000, help="Write CSV every N row-updates")
	ap.add_argument("--max_tokens", type=int, default=128, help="Max completion tokens per request")
	ap.add_argument("--cache_path", default="cache/", help="Cache dir used by caching_and_prompting")
	args = ap.parse_args()

	if not os.path.exists(args.input):
		print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
		return 2

	df = pd.read_csv(args.input)

	if "ai_best_score" not in df.columns:
		print("ERROR: input CSV must contain column 'ai_best_score'", file=sys.stderr)
		print(f"Columns found: {list(df.columns)}", file=sys.stderr)
		return 2

	out_cols = [
		"is_coding_llm",
		"is_coding_llm_explanation",
		"has_any_bias_llm",
		"has_any_bias_llm_explanation",
		"llm_model",
		"llm_threshold",
		"llm_timestamp_utc",
		"bias_cue_type_llm",
		"bias_cue_type_llm_explanation",
	]
	for c in out_cols:
		if c not in df.columns:
			df[c] = pd.NA

	eligible = df["ai_best_score"].fillna(-1.0) >= float(args.threshold)
	idxs = list(df.index[eligible])

	if args.limit and args.limit > 0:
		idxs = idxs[: args.limit]

	if args.resume:
		idxs = [i for i in idxs if pd.isna(df.at[i, "is_coding_llm"])]

	if not idxs:
		df.to_csv(args.output, index=False)
		print("Nothing to do (no eligible rows after --resume/--limit).")
		return 0

	ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

	# ---------------------------
	# Stage A: build ALL prompts
	# ---------------------------
	stage_a_prompts = []
	stage_a_prompt_texts = []  # keep prompt_text for later reuse
	for idx in idxs:
		prompt_text = pick_prompt_text(df.loc[idx])
		stage_a_prompt_texts.append(prompt_text)
		stage_a_prompts.append(
			"Classify the following user prompt.\n\n"
			"PROMPT:\n"
			f"{prompt_text}\n\n"
			"Remember: output explanation first, then on the final line output TRUE or FALSE only."
		)

	# One call, no chunking
	a_outs = instruct_model(
		stage_a_prompts,
		model=args.model,
		system_instructions=[CODING_SYSTEM] * len(stage_a_prompts),
		temperature=0.0,
		max_tokens=args.max_tokens,
		cache_path=args.cache_path,
		# You can optionally pass stop=["\nTRUE", "\nFALSE"] but leaving off to avoid truncation surprises.
	)

	# Apply Stage A results
	updated = 0
	pbar_a = tqdm(range(len(idxs)), desc=f"Applying Stage A (ai_best_score>={args.threshold})", unit="row")
	for j in pbar_a:
		idx = idxs[j]
		a_out = a_outs[j]
		a_expl, a_bool = parse_explanation_and_bool(a_out)

		if a_bool is None:
			a_bool = False
			if not a_expl:
				a_expl = "Parsing failed; defaulted to FALSE (conservative)."

		df.at[idx, "is_coding_llm"] = bool(a_bool)
		df.at[idx, "is_coding_llm_explanation"] = a_expl
		df.at[idx, "llm_model"] = args.model
		df.at[idx, "llm_threshold"] = float(args.threshold)
		df.at[idx, "llm_timestamp_utc"] = ts

		if not a_bool:
			df.at[idx, "has_any_bias_llm"] = pd.NA
			df.at[idx, "has_any_bias_llm_explanation"] = pd.NA
			df.at[idx, "bias_cue_type_llm"] = pd.NA
			df.at[idx, "bias_cue_type_llm_explanation"] = pd.NA

		updated += 1
		if args.checkpoint_every and updated % args.checkpoint_every == 0:
			df.to_csv(args.output, index=False)
			pbar_a.write(f"[checkpoint] wrote {args.output} (row-updates={updated})")

	# -----------------------------------------
	# Stage B: ONLY for coding=True rows (ALL)
	# -----------------------------------------
	stage_b_idxs = [idx for idx in idxs if bool(df.at[idx, "is_coding_llm"]) is True]

	if stage_b_idxs:
		stage_b_prompts = []
		# reuse the already-extracted prompt_texts for speed
		idx_to_prompt_text = {idxs[k]: stage_a_prompt_texts[k] for k in range(len(idxs))}
		for idx in stage_b_idxs:
			prompt_text = idx_to_prompt_text[idx]
			stage_b_prompts.append(
				"Check the following coding-related user prompt for explicit bias-inducing linguistic cues.\n\n"
				"PROMPT:\n"
				f"{prompt_text}\n\n"
				"Remember: output explanation first, then on the final line output TRUE or FALSE only."
			)

		# One call, no chunking
		b_outs = instruct_model(
			stage_b_prompts,
			model=args.model,
			system_instructions=[BIAS_SYSTEM] * len(stage_b_prompts),
			temperature=0.0,
			# max_tokens=args.max_tokens,
			cache_path=args.cache_path,
			reasoning_effort='default',
		)

		pbar_b = tqdm(range(len(stage_b_idxs)), desc="Applying Stage B (bias)", unit="row")
		for j in pbar_b:
			idx = stage_b_idxs[j]
			b_out = b_outs[j]
			b_expl, b_bool = parse_explanation_and_bool(b_out)

			if b_bool is None:
				b_bool = False
				if not b_expl:
					b_expl = "Parsing failed; defaulted to FALSE (conservative)."

			df.at[idx, "has_any_bias_llm"] = bool(b_bool)
			df.at[idx, "has_any_bias_llm_explanation"] = b_expl

			updated += 1
			if args.checkpoint_every and updated % args.checkpoint_every == 0:
				df.to_csv(args.output, index=False)
				pbar_b.write(f"[checkpoint] wrote {args.output} (row-updates={updated})")

	# Stage C: for bias=True rows, pick which type (1..8) or 9=no bias
	stage_c_idxs = [idx for idx in stage_b_idxs if bool(df.at[idx, "has_any_bias_llm"]) is True]

	# Default label=9 for coding prompts where Stage B says no bias (so column is always filled for coding prompts)
	for idx in stage_b_idxs:
		if bool(df.at[idx, "has_any_bias_llm"]) is False:
			df.at[idx, "bias_cue_type_llm"] = BIAS_CUE_LABELS[9]
			df.at[idx, "bias_cue_type_llm_explanation"] = "No bias cue detected (Stage B FALSE)."

	if stage_c_idxs:
		stage_c_prompts = []
		for idx in stage_c_idxs:
			prompt_text = idx_to_prompt_text[idx]  # reuse your dict from idx -> prompt_text
			stage_c_prompts.append(
				"Choose the single best bias cue type for the following coding-related prompt.\n\n"
				"PROMPT:\n"
				f"{prompt_text}\n\n"
				"Remember: explanation first, then on the final line output one integer 1–9 only."
			)

		c_outs = instruct_model(
			stage_c_prompts,
			model=args.model,
			system_instructions=[BIAS_TYPE_SYSTEM] * len(stage_c_prompts),
			temperature=0.0,
			# max_tokens=args.max_tokens,
			cache_path=args.cache_path,
			reasoning_effort='default',
		)

		for j, idx in enumerate(stage_c_idxs):
			c_expl, c_label = parse_explanation_and_label_1to9(c_outs[j])
			if c_label is None or not (1 <= c_label <= 9):
				c_label = 9
				if not c_expl:
					c_expl = "Parsing failed; defaulted to 9 (no bias)."
			df.at[idx, "bias_cue_type_llm"] = BIAS_CUE_LABELS[int(c_label)]
			df.at[idx, "bias_cue_type_llm_explanation"] = c_expl


	df.to_csv(args.output, index=False)
	print(
		f"Done. Eligible processed={len(idxs)}; coding_true={len(stage_b_idxs)}. "
		f"Wrote: {args.output}"
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
