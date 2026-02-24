#!/usr/bin/env python3
"""
Compute cue-span alignment prevalence between DevGPT positive entries and PROBE-SWE.

Edits in this version (per request):
- Enforce single-candidate prompting: cand_batch_size is hard-set to 1 and the CLI flag is removed.
- ELIMINATED any probe lookup / late attachment: probe_biased_prompt is attached at candidate construction time
  and carried through to all_outputs.csv and top100_for_manual_validation_ANNOTATE.csv.

- Candidate generation: TF-IDF (word + char ngrams) over PROBE cue spans (per bias).
- Scoring: Groq qwen/qwen3-32b via `from lib import instruct_model`.
- Batching: each prompt scores EXACTLY ONE candidate (idx=0) per DevGPT entry per candidate-rank "batch_no".
- Outputs:
    * all_outputs.csv                       (one row per (DevGPT entry, PROBE candidate) scored)
    * best_per_devgpt.csv                   (best-scoring candidate per DevGPT entry)
    * prevalence_by_bias.csv                (DevGPT-level: any aligned candidate >= threshold)
    * prevalence_by_bias_probeswe.csv       (Candidate-level: fraction of candidate rows aligned)
    * top100_for_manual_validation_ANNOTATE.csv (top 100 aligned candidate rows + text context + blank human cols)

Usage:
  export GROQ_API_KEY="..."
  python compute_cue_alignment_tfidf_single_candidate.py \
      --dev_csv "/path/DevGPT positive entries.csv" \
      --probe_path "/path/PROBE-SWE dataset.zip" \
      --bias_col "probe_best_bias" \
      --text_col "prompt_clean" \
      --topk 250 \
      --threshold 0.60 \
      --out_dir "out_alignment"

Notes:
- The scorer is conservative: surface-form cue alignment only.
- Hindsight: self-correction vs outcome-known narration should be weak/none unless surface markers match.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(str(Path(__file__).resolve().parent))

from lib import instruct_model, set_deterministic


BIAS_KEYS = [
    "confirmation bias",
    "framing effect",
    "overconfidence bias",
    "hyperbolic discounting",
    "anchoring bias",
    "bandwagon effect",
    "availability bias",
    "hindsight bias",
]

# Table exemplars (from Table \\ref{tab:devgpt_coding_cues})
TABLE_EXAMPLES = {
    "confirmation bias": {
        "dev": "We don't need that second if, because ... Am I right?",
        "probe": "... keeping passwords in plain text ... [is] so much easier to manage, right?",
        "note": "Tag-question / confirmation-seeking cue.",
    },
    "framing effect": {
        "dev": "Add more exciting css too",
        "probe": "... but isn’t it more exciting to release now and worry later?",
        "note": "Framing via evaluative adjective.",
    },
    "overconfidence bias": {
        "dev": "... im almost sure i need to add a library, right?",
        "probe": "... I feel confident that there won't be any unforeseen issues ...",
        "note": "Overconfidence markers (sure/confident).",
    },
    "hyperbolic discounting": {
        "dev": "... eliminate the screen size watching for now",
        "probe": "... I think it’s better to avoid disruption for now.",
        "note": "Present-focus / deferral markers (for now).",
    },
    "anchoring bias": {
        "dev": "This is not needed. The previous version was fine...",
        "probe": "I remember that in a previous project, the team didn’t use a ...",
        "note": "Reference to prior baseline / previous version.",
    },
    "bandwagon effect": {
        "dev": "... it looks like linregress is popular, implement the above using it.",
        "probe": "... which seems to be a popular choice lately",
        "note": "Popularity/consensus cue (popular choice).",
    },
    "availability bias": {
        "dev": "I read somewhere that ...",
        "probe": "I remember reading somewhere that ...",
        "note": "Vague recall/source cue.",
    },
    "hindsight bias": {
        "dev": "i should have mentioned that the reset should come before the mode set...",
        "probe": "Shortly after production deployment...",
        "note": "Weak alignment example (self-correction vs outcome-known narrative).",
    },
}


def _table_example_block(bias: str) -> str:
    ex = TABLE_EXAMPLES.get(bias)
    if not ex:
        return ""
    return (
        "TABLE_EXAMPLE (calibration anchor):\n"
        f"- DevGPT: \"{ex['dev']}\"\n"
        f"- PROBE: \"{ex['probe']}\"\n"
        f"- Note: \"{ex['note']}\"\n"
    )


def normalize_bias(b: str) -> str:
    """
    Normalize bias labels WITHOUT duplicating already-canonical names.
    """
    if not isinstance(b, str):
        return ""
    b = re.sub(r"\s+", " ", b.strip().lower())
    if b in BIAS_KEYS:
        return b

    # short forms / variants
    if b in {"time preference", "present bias"}:
        return "hyperbolic discounting"
    if b == "bandwagon":
        return "bandwagon effect"
    if b == "availability":
        return "availability bias"
    if b == "overconfidence":
        return "overconfidence bias"
    if b == "anchoring":
        return "anchoring bias"
    if b == "confirmation":
        return "confirmation bias"
    if b == "hindsight":
        return "hindsight bias"
    if b == "framing":
        return "framing effect"
    return b


def extract_probe_cue_span_words(unbiased: str, biased: str, max_words: int = 50) -> str:
    """
    Minimal-ish cue span based on word-level diff. Approximates "minimal insertion/rewrite".
    """
    uw = (unbiased or "").split()
    bw = (biased or "").split()
    sm = difflib.SequenceMatcher(None, uw, bw)
    ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
    if not ops:
        return ""
    # choose op with largest changed segment in biased
    _, _, _, j1, j2 = max(ops, key=lambda op: (op[4] - op[3]))
    span_words = bw[j1:j2]
    if len(span_words) > max_words:
        span_words = span_words[:max_words]
    span = " ".join(span_words)
    span = re.sub(r"\s+", " ", span).strip()
    return span


@dataclass
class ProbeCue:
    bias: str
    probe_pair_id: str
    cue_span: str
    biased_prompt: str


def iter_probe_dataset_specs(probe_path: Path) -> List[Tuple[str, Tuple]]:
    """
    Discover PROBE-SWE jsons matching "*_dataset_*.json".

    probe_path may be:
      - a .zip file (members are searched)
      - a directory (searched recursively)
      - a single json file
    Returns: list of (source_name, spec_tuple)
      spec_tuple is either ("zip", zip_fp, member_name) or ("file", fp)
    """
    probe_path = Path(probe_path)

    # Zip case
    if probe_path.is_file() and zipfile.is_zipfile(probe_path):
        with zipfile.ZipFile(probe_path, "r") as z:
            members = [
                n
                for n in z.namelist()
                if re.search(r"_dataset_.*\.json$", Path(n).name)
                and not n.startswith("__MACOSX/")
                and "/._" not in n
                and not Path(n).name.startswith("._")
            ]
        return [(Path(n).name, ("zip", str(probe_path), n)) for n in sorted(members)]

    # Directory case
    if probe_path.is_dir():
        files = sorted([p for p in probe_path.rglob("*.json") if re.search(r"_dataset_.*\.json$", p.name)])
        return [(p.name, ("file", str(p))) for p in files]

    # Single json fallback
    if probe_path.is_file() and probe_path.suffix.lower() == ".json":
        return [(probe_path.name, ("file", str(probe_path)))]

    raise FileNotFoundError(f"probe_path must be a zip, directory, or json file: {probe_path}")


def load_probe_cues(probe_path: Path) -> Dict[str, List[ProbeCue]]:
    sources = iter_probe_dataset_specs(probe_path)
    if not sources:
        raise FileNotFoundError(f"No *_dataset_*.json files found in {probe_path}")

    cues: Dict[str, List[ProbeCue]] = {b: [] for b in BIAS_KEYS}

    for source_name, spec in sources:
        kind = spec[0]
        if kind == "zip":
            _, zip_fp, member = spec
            with zipfile.ZipFile(zip_fp, "r") as z:
                with z.open(member) as f:
                    probe = json.load(f)
        else:
            _, fp = spec
            with open(fp, "r", encoding="utf-8") as f:
                probe = json.load(f)

        if not isinstance(probe, dict):
            continue

        for bias, items in probe.items():
            nb = normalize_bias(bias)
            if nb not in BIAS_KEYS:
                continue
            if not isinstance(items, list):
                continue

            for idx, it in enumerate(items):
                if not isinstance(it, dict):
                    continue
                cue = extract_probe_cue_span_words(it.get("unbiased", ""), it.get("biased", ""))
                pair_raw = it.get("pair", None)

                # Unique across multiple json sources
                if pair_raw is None:
                    pair_id = f"{Path(source_name).stem}:{idx}"
                else:
                    pair_id = f"{Path(source_name).stem}:{pair_raw}"

                cues[nb].append(
                    ProbeCue(
                        bias=nb,
                        probe_pair_id=str(pair_id),
                        cue_span=cue,
                        biased_prompt=it.get("biased", ""),
                    )
                )

    cues = {b: v for b, v in cues.items() if v}
    return cues


def build_indexes(cues_by_bias: Dict[str, List[ProbeCue]]):
    """
    Returns per-bias vectorizers and TF-IDF matrices for two channels:
      - word ngrams
      - char ngrams
    """
    idx = {}
    for bias, cues in cues_by_bias.items():
        texts = [c.cue_span for c in cues]
        v_word = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        X_word = v_word.fit_transform(texts)

        v_char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X_char = v_char.fit_transform(texts)

        idx[bias] = dict(
            cues=cues,
            v_word=v_word,
            X_word=X_word,
            v_char=v_char,
            X_char=X_char,
        )
    return idx


def topk_candidates(dev_text: str, index_obj: dict, topk: int = 25) -> List[Tuple[int, float]]:
    """
    Return list of (candidate_index, tfidf_score) sorted descending by score.
    """
    cues = index_obj["cues"]
    if not dev_text:
        return []
    q_word = index_obj["v_word"].transform([dev_text])
    q_char = index_obj["v_char"].transform([dev_text])

    s_word = (q_word @ index_obj["X_word"].T).toarray().ravel()
    s_char = (q_char @ index_obj["X_char"].T).toarray().ravel()
    s = (s_word + s_char) / 2.0

    k1 = min(topk, len(cues))
    if k1 <= 0:
        return []
    top_idx = np.argpartition(-s, range(k1))[:k1]
    top_idx = top_idx[np.argsort(-s[top_idx])]
    return [(int(i), float(s[i])) for i in top_idx]


SYSTEM = """You are a strict cue-span alignment scorer for ecological plausibility checks.

Goal:
Given (A) a DevGPT excerpt and (B) multiple candidate PROBE-SWE cue spans for the SAME bias type,
identify which PROBE cues are linguistically analogous to the DevGPT cue(s).

You will be shown one TABLE_EXAMPLE for the current bias type.
It contains:
- a short DevGPT excerpt with the key cue phrase highlighted conceptually (no literal markup needed), and
- an analogous PROBE-SWE cue span.
Use this TABLE_EXAMPLE ONLY as an example of cue alignment.
Score based on similarity between the ACTUAL DevGPT text and the ACTUAL candidate PROBE cue span.

Strength rubric:
- strong: same cue function in the same discourse role, even if phrasing differs (i.e., clear paraphrase OR functional equivalence)
- weak: similar discourse role but noticeably different surface construction; borderline plausibility
- none: no meaningful surface-marker alignment

Output requirement:
Return JSON only (no extra prose). Follow the schema given in the user prompt exactly.
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def clamp_span_to_text(span: str, text: str) -> str:
    """
    Ensure dev_cue_span is a verbatim substring of text.
    """
    span = (span or "").strip().strip('…').strip('.').strip()
    if not span:
        return ""
    if span in text:
        return span
    m = re.search(re.escape(span), text, flags=re.IGNORECASE)
    if m:
        return text[m.start() : m.end()]
    return ""


def parse_json_loose(s: Optional[str]) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    m = _JSON_RE.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def make_alignment_prompt(
    bias: str,
    dev_id: Any,
    dev_text: str,
    cands: List[ProbeCue],
    cand_scores: List[float],
    cand_max_chars: int = 140,
    include_ctx_for_short_spans: bool = True,
) -> str:
    """
    Build a compact prompt for a SINGLE candidate (enforced by caller), asking to score it.
    The model must keep probe_pair_id values exactly as provided.
    """
    example_block = _table_example_block(bias).strip()

    lines = []
    for j, (c, sc) in enumerate(zip(cands, cand_scores)):
        cue = (c.cue_span or "").strip()
        if include_ctx_for_short_spans and len(cue) < 10:
            ctx = re.sub(r"\s+", " ", c.biased_prompt or "")[:160]
            cue = f"{cue} | ctx: {ctx}"
        if cand_max_chars and len(cue) > cand_max_chars:
            cue = cue[:cand_max_chars].rstrip() + "…"
        lines.append(f"- idx={j} probe_pair_id={c.probe_pair_id} tfidf={sc:.4f}: {cue}")
    cand_block = "\n".join(lines)

    return f"""BIAS_TYPE: {bias}
DEV_ID: {dev_id}
DEVGPT_TEXT:
\"\"\"{dev_text}\"\"\"

{example_block}

CANDIDATE_PROBE_CUE_SPANS (score each candidate independently; keep probe_pair_id exactly):
{cand_block}

Task:
For EACH candidate in this batch:
1) Decide whether it has an analogous cue span to DevGPT (surface-form alignment).
2) If aligned=true, extract:
   - dev_cue_span: MUST be an EXACT verbatim substring from DEVGPT_TEXT (or "" if none).
   - probe_cue_span: minimal cue span from the candidate line (or "").
3) Assign a conservative score in [0,1].
4) If aligned=false, set alignment_strength="none", score=0.0, dev_cue_span="", probe_cue_span="".

Output JSON schema (JSON only; no prose):
{{
  "dev_id": <same as DEV_ID>,
  "bias_type": "<same as BIAS_TYPE>",
  "candidates": [
    {{
      "idx": 0,
      "probe_pair_id": "<string exactly as provided>",
      "aligned": true/false,
      "alignment_strength": "strong"|"weak"|"none",
      "score": 0.0-1.0,
      "dev_cue_span": "<verbatim substring or empty>",
      "probe_cue_span": "<string or empty>",
      "notes": "<<=12 words>"
    }}
  ]
}}
"""


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return float(center - half), float(center + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_csv", type=Path, required=True)
    ap.add_argument("--probe_path", type=Path, required=True)
    ap.add_argument("--bias_col", type=str, default="probe_best_bias")
    ap.add_argument("--text_col", type=str, default="prompt_clean")

    ap.add_argument("--model", type=str, default="qwen/qwen3-32b")
    ap.add_argument("--reasoning_effort", type=str, default="none")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=1200)

    ap.add_argument("--topk", type=int, default=250)
    # cand_batch_size is intentionally removed; single-candidate prompting is enforced.
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cand_max_chars", type=int, default=140)
    ap.add_argument("--out_dir", type=Path, default=Path("out_alignment"))
    args = ap.parse_args()

    set_deterministic(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dev = pd.read_csv(args.dev_csv)

    # robust dev_id
    if "id" in dev.columns:
        dev["dev_id"] = dev["id"]
    else:
        dev["dev_id"] = pd.NA
    dev["dev_id"] = dev["dev_id"].fillna(dev.index.to_series()).astype(str)

    dev["bias_norm"] = dev[args.bias_col].apply(normalize_bias)
    dev["text"] = dev[args.text_col].fillna("").astype(str)

    # sanity
    dev_bias_counts = dev["bias_norm"].value_counts(dropna=False).to_dict()
    print("[dev] bias_norm counts:", dev_bias_counts)
    dev_unknown = sorted(set(dev_bias_counts.keys()) - set(BIAS_KEYS) - {""})
    if dev_unknown:
        print("[dev][WARN] unrecognized bias labels after normalization:", dev_unknown)

    cues_by_bias = load_probe_cues(args.probe_path)
    probe_counts = {b: len(v) for b, v in cues_by_bias.items()}
    print("[probe] cue counts by bias:", probe_counts)
    if len([b for b, c in probe_counts.items() if c > 0]) <= 1:
        print("[probe][WARN] Only <=1 bias loaded from PROBE. Check normalize_bias() and *_dataset_*.json contents.")

    index = build_indexes(cues_by_bias)

    # Precompute candidates per DevGPT row (key by dev_row)
    dev_rows: List[Tuple[int, str, str, str]] = []
    dev_cands: Dict[int, List[Tuple[int, float]]] = {}  # dev_row -> list[(cand_idx, tfidf)]

    for dev_row, row in dev.iterrows():
        bias = row["bias_norm"]
        if bias not in index:
            continue
        dev_id = row["dev_id"]
        dev_text = row["text"]

        cand = topk_candidates(dev_text, index[bias], topk=args.topk)  # list[(idx, tfidf)]
        dev_rows.append((int(dev_row), dev_id, bias, dev_text))
        dev_cands[int(dev_row)] = cand

    max_batches = max((len(v) for v in dev_cands.values()), default=0)
    print("[run] dev_rows:", len(dev_rows), "max_candidates:", max_batches)

    out_rows: List[Dict[str, Any]] = []

    # Each batch_no corresponds to candidate rank. Each prompt scores exactly ONE candidate (idx=0).
    for batch_no in range(max_batches):
        prompts: List[str] = []
        metas: List[Tuple[int, str, str, str, int, int, float, ProbeCue]] = []
        # metas: (dev_row, dev_id, bias, dev_text, batch_no, cand_i, tfidf_sc, cue_obj)

        for dev_row, dev_id, bias, dev_text in dev_rows:
            cand_list = dev_cands.get(dev_row, [])
            if batch_no >= len(cand_list):
                continue

            cand_i, tfidf_sc = cand_list[batch_no]
            cue_obj = index[bias]["cues"][cand_i]

            prompts.append(
                make_alignment_prompt(
                    bias=bias,
                    dev_id=dev_id,
                    dev_text=dev_text,
                    cands=[cue_obj],  # SINGLE candidate
                    cand_scores=[tfidf_sc],
                    cand_max_chars=args.cand_max_chars,
                    include_ctx_for_short_spans=True,
                )
            )
            metas.append((dev_row, dev_id, bias, dev_text, batch_no, cand_i, tfidf_sc, cue_obj))

        if not prompts:
            continue

        outs = instruct_model(
            prompts,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_instructions=[SYSTEM] * len(prompts),
            include_reasoning=False,
        )

        for out, meta in zip(outs, metas):
            dev_row, dev_id, bias, dev_text, batch_no, cand_i, tfidf_sc, cue_obj = meta
            obj = parse_json_loose(out) or {}
            cand_out = obj.get("candidates", [])
            if not isinstance(cand_out, list):
                cand_out = []

            # Expect single candidate with idx=0; fallback to empty dict
            it: Dict[str, Any] = {}
            for x in cand_out:
                if isinstance(x, dict) and str(x.get("idx", "")) == "0":
                    it = x
                    break

            aligned = bool(it.get("aligned", False))
            strength = (it.get("alignment_strength", "none") or "none").strip().lower()
            if strength not in {"strong", "weak", "none"}:
                strength = "none"

            score = it.get("score", 0.0)
            try:
                score = float(score)
            except Exception:
                score = float("nan")

            raw_dev_span = (it.get("dev_cue_span") or "").strip()
            dev_span = clamp_span_to_text(raw_dev_span, dev_text)

            # hard enforcement: non-verbatim => invalidate
            if raw_dev_span and not dev_span:
                aligned = False
                strength = "none"
                score = 0.0

            if not aligned:
                strength = "none"
                score = 0.0
                dev_span = ""

            out_rows.append(
                {
                    "dev_row": int(dev_row),
                    "dev_id": dev_id,
                    "bias_norm": bias,
                    "batch_no": int(batch_no),

                    # With single-candidate prompting:
                    "cand_local_idx": 0,
                    "cand_global_rank": int(batch_no),
                    "tfidf_score": float(tfidf_sc),

                    "probe_pair_id": cue_obj.probe_pair_id,
                    "probe_candidate_span": (cue_obj.cue_span or "").strip(),
                    # Attached at construction time (no probe_lookup indirection)
                    "probe_biased_prompt": (cue_obj.biased_prompt or "").strip(),

                    "aligned_pred": int(aligned),
                    "alignment_strength": strength,
                    "alignment_score": score,
                    "dev_cue_span": dev_span,
                    "probe_cue_span": (it.get("probe_cue_span") or "").strip(),
                    "notes": (it.get("notes") or "").strip(),

                    "raw_model_output": out,
                }
            )

        print(f"[cand_rank {batch_no+1}/{max_batches}] scored {len(prompts)} prompts; total_rows={len(out_rows)}")

    all_df = pd.DataFrame(out_rows)
    all_df.to_csv(args.out_dir / "all_outputs.csv", index=False)

    # flags
    all_df["is_aligned"] = (all_df["aligned_pred"] == 1) & (all_df["alignment_score"] >= args.threshold)
    all_df["is_strong_aligned"] = all_df["is_aligned"] & (all_df["alignment_strength"] == "strong")

    # Best per DevGPT (by alignment_score; tie-break strong>weak>none; then tfidf)
    def _strength_rank(x: str) -> int:
        return {"strong": 2, "weak": 1, "none": 0}.get(str(x), 0)

    tmp = all_df.copy()
    tmp["strength_rank"] = tmp["alignment_strength"].map(_strength_rank)
    tmp = tmp.sort_values(
        ["dev_id", "alignment_score", "strength_rank", "tfidf_score"],
        ascending=[True, False, False, False],
    )
    best = tmp.groupby("dev_id", as_index=False).head(1).drop(columns=["strength_rank"])
    best.to_csv(args.out_dir / "best_per_devgpt.csv", index=False)

    # Prevalence by bias (DevGPT-level; any aligned candidate)
    prev_rows = []
    for bias in sorted(set(all_df["bias_norm"]) & set(index.keys())):
        sub = all_df[all_df["bias_norm"] == bias]
        g = sub.groupby("dev_id", dropna=False)

        aligned_any = g["is_aligned"].any()
        strong_any = g["is_strong_aligned"].any()
        weak_any = aligned_any & ~strong_any

        n = int(aligned_any.shape[0])
        k = int(aligned_any.sum())

        p = (k / n) if n else np.nan
        lo, hi = wilson_ci(k, n)

        prev_rows.append(
            dict(
                bias=bias,
                n_devgpt=n,
                aligned_k=k,
                prevalence=p,
                wilson95_lo=lo,
                wilson95_hi=hi,
                strong_k=int(strong_any.sum()),
                weak_k=int(weak_any.sum()),
            )
        )
    prev = pd.DataFrame(prev_rows).sort_values(["prevalence", "n_devgpt"], ascending=[False, False])
    prev.to_csv(args.out_dir / "prevalence_by_bias.csv", index=False)

    # Candidate-level prevalence (over all scored candidate rows)
    prev_rows_probe = []
    for bias in sorted(set(all_df["bias_norm"]) & set(index.keys())):
        sub = all_df[all_df["bias_norm"] == bias]
        n = int(len(sub))
        k = int(sub["is_aligned"].sum())
        p = (k / n) if n else np.nan
        lo, hi = wilson_ci(k, n)

        prev_rows_probe.append(
            dict(
                bias=bias,
                n_probeswe=n,
                aligned_k=k,
                prevalence=p,
                wilson95_lo=lo,
                wilson95_hi=hi,
                strong_k=int(sub["is_strong_aligned"].sum()),
                weak_k=int(((sub["alignment_strength"] == "weak") & sub["is_aligned"]).sum()),
            )
        )
    prev_probe = pd.DataFrame(prev_rows_probe).sort_values(["prevalence", "n_probeswe"], ascending=[False, False])
    prev_probe.to_csv(args.out_dir / "prevalence_by_bias_probeswe.csv", index=False)

    # Top100 for manual validation (UNIQUE dev_id)
    top = all_df[all_df["is_aligned"]].copy()

    # tie-break: score desc, strong>weak>none, then tfidf desc
    _strength_rank_map = {"strong": 2, "weak": 1, "none": 0}
    top["strength_rank"] = top["alignment_strength"].map(lambda x: _strength_rank_map.get(str(x), 0))

    top = top.sort_values(
        ["dev_id", "alignment_score", "strength_rank", "tfidf_score"],
        ascending=[True, False, False, False],
    )

    # keep the highest-scoring row per dev_id (stable tie-break with tfidf_score)
    top = (
        top.sort_values(["dev_id", "alignment_score", "tfidf_score"], ascending=[True, False, False])
           .groupby("dev_id", as_index=False)
           .head(1)
           .sort_values(["alignment_score", "tfidf_score"], ascending=[False, False])
           .head(100)
    )

    dev_text_map = dev.set_index("dev_id")["text"].to_dict()
    top["dev_text"] = top["dev_id"].map(dev_text_map).fillna("")
    top["probe_text"] = top["probe_biased_prompt"]
    del top["probe_biased_prompt"]

    # probe_biased_prompt is already present from construction-time attachment
    top["human_aligned"] = ""
    top["human_notes"] = ""
    top.to_csv(args.out_dir / "top100_for_manual_validation_ANNOTATE.csv", index=False)

    print(f"Wrote outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
