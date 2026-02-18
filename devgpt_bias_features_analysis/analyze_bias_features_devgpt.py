import argparse
import os
import sys
import re
import json
import csv
import zipfile
import pathlib
import difflib
import math
import random
import hashlib
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm

# Increase CSV field size limit to support very large text cells in datasets/annotations.
def _set_csv_field_size_limit():
	max_int = getattr(sys, "maxsize", 2**31 - 1)
	while True:
		try:
			csv.field_size_limit(max_int)
			return max_int
		except OverflowError:
			max_int = int(max_int / 10)

_set_csv_field_size_limit()

# ------------------------- Caching helpers (shared with lib.py when available) -------------------------

def _try_import_lib_cache():
	"""Try to import caching helpers from lib.py (preferred)."""
	try:
		from lib import load_or_create_cache, get_cached_values, create_cache  # type: ignore
		return load_or_create_cache, get_cached_values, create_cache, True
	except Exception:
		return None, None, None, False

def _fallback_create_cache(file_name, create_fn):
	os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
	result = create_fn()
	with open(file_name, "wb") as f:
		pickle.dump(result, f)
	return result

def _fallback_load_cache(file_name):
	if os.path.isfile(file_name):
		with open(file_name, "rb") as f:
			return pickle.load(f)
	return None

def _fallback_load_or_create_cache(file_name, create_fn):
	result = _fallback_load_cache(file_name)
	if result is None:
		result = _fallback_create_cache(file_name, create_fn)
	return result

def _fallback_get_cached_values(value_list, cache, fetch_fn, cache_name=None, key_fn=lambda x: x, empty_is_missing=True, transform_fn=None, **_kwargs):
	"""Fallback version of lib.get_cached_values (safe for numpy arrays)."""
	missing = []
	seen = set()
	for q in value_list:
		if not q:
			continue
		k = key_fn(q)
		if k in seen:
			continue
		seen.add(k)
		if k not in cache:
			missing.append(q)
		elif empty_is_missing:
			v = cache.get(k)
			if v is None or v == "":
				missing.append(q)
	if missing:
		for q, v in fetch_fn(tuple(missing)):
			cache[key_fn(q)] = v
		if cache_name:
			_fallback_create_cache(cache_name, lambda: cache)
	out = [cache.get(key_fn(q)) if q else None for q in value_list]
	if transform_fn:
		out = list(map(transform_fn, out))
	return out

_load_or_create_cache, _get_cached_values, _create_cache, _HAVE_LIB_CACHE = _try_import_lib_cache()
if not _HAVE_LIB_CACHE:
	_load_or_create_cache = _fallback_load_or_create_cache
	_get_cached_values = _fallback_get_cached_values
	_create_cache = _fallback_create_cache

# Disk embedding cache directory (set in main; falls back to ./cache_embeddings)
_SBERT_EMB_CACHE_DIR = os.environ.get("SBERT_EMB_CACHE_DIR", "")
_SBERT_EMB_DISK_CACHES = {}

# Disk HF "Processed data" prediction cache directory (set in main; falls back to ~/.cache/bias_features_devgpt/hf_pred_cache)
_HF_PRED_CACHE_DIR = os.environ.get("HF_PRED_CACHE_DIR", "")
_HF_PRED_DISK_CACHES = {}

# ------------------------- Preprocessing -------------------------

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")
WS_RE = re.compile(r"\s+")

CODELIKE_RE = re.compile(
	r"^\s*(?:[{(\[]|#include|using\s+|import\s+|from\s+|def\s+|class\s+|public\s+|private\s+|protected\s+|fn\s+|let\s+|const\s+|var\s+|if\s*\(|for\s*\(|while\s*\(|switch\s*\(|try\s*:|catch\s*\(|return\s+)",
	re.I,
)
SYMBOL_DENSITY_RE = re.compile(r"[^A-Za-z0-9\s]")

def strip_code(text: str):
	"""
	Removes fenced code blocks, inline code, and strongly code-like lines.
	This keeps the analysis focused on natural language cues.
	"""
	if not text:
		return ""
	t = FENCE_RE.sub(" ", text)
	t = INLINE_CODE_RE.sub(" ", t)
	lines = []
	for line in t.splitlines():
		raw = line.rstrip("\n")
		if not raw.strip():
			continue
		# Drop comment-only / code-comment lines
		if re.match(r"^\s*(#|//|/\*|\*)\s*", raw):
			continue
		# drop very code-like lines
		sym = len(SYMBOL_DENSITY_RE.findall(raw))
		if CODELIKE_RE.search(raw) or (sym / max(1, len(raw)) > 0.20):
			continue
		lines.append(raw)
	t = " ".join(lines)
	t = WS_RE.sub(" ", t).strip()
	return t

DECISION_RE = re.compile(r"\b(vs\.?|versus|option\s+[ab]|choose|which|recommend|should\s+i|better|best|prefer|trade-?off|pros\s+and\s+cons)\b", re.I)

def is_decision_prompt(text: str):
	return bool(DECISION_RE.search(text or ""))

# ------------------------- Codebook (baseline, keep for transparency) -------------------------

def compile_codebook():
	"""
	Transparent, reviewer-friendly regex codebook (baseline).
	Keep this even when using AI methods.
	"""
	cb = {
		"confirmation_cues": {
			"strong": [
				r"\b(can you|could you|please)\s+(confirm|validate|double\s*check)\b",
				r"\bI\s+(think|believe|suspect)\b.*\b(confirm|validate)\b",
				r"\bI\s+(already|still)\s+(know|think)\b.*\b(right|correct)\b",
				r"\bisn['’]t\s+it\s+true\s+that\b",
				r"\bprove\s+that\b",
			],
			"weak": [
				r"\bconfirm\b",
				r"\bvalidate\b",
				r"\bdouble\s*check\b",
				r"\bdoes\s+this\s+mean\b",
			],
		},
		"anchoring_phrases": {
			"strong": [
				r"\b(baseline|anchor|starting\s+point|reference\s+point)\b",
				r"\bI\s+(heard|was\s+told|read)\s+it\s+(takes|costs)\s+\$?\d+",
				r"\b(around|about|roughly|approx(?:\.|imately)?)\s+\$?\d+(\.\d+)?\b",
				r"\b(around|about|roughly|approx(?:\.|imately)?)\s+\d+\s*(hours?|days?|weeks?|months?)\b",
				r"\b(in\s+line\s+with|based\s+on)\s+(a|the)\s+(estimate|guess|number)\b",
			],
			"weak": [
				r"\bbaseline\b",
				r"\bestimate\b",
				r"\broughly\b",
				r"\bapprox(?:\.|imately)?\b",
				r"\babout\s+\d+\b",
			],
		},
		"framing_language": {
			"strong": [
				r"\b(waste|pointless|useless|painful|drudgery|mess|nightmare)\b",
				r"\b(best\s+case|worst\s+case)\b",
				r"\b(don['’]t\s+want\s+to\s+lose|avoid\s+losing)\b",
				r"\b(win|lose)\b.*\bby\s+choosing\b",
			],
			"weak": [
				r"\bbetter\b",
				r"\bworse\b",
				r"\bideal\b",
				r"\bavoid\b",
				r"\brisk\b",
			],
		},
		"bandwagon_cues": {
			"strong": [
				r"\b(everyone|most\s+people|most\s+teams|many\s+teams)\s+(use|prefer|recommend)\b",
				r"\b(industry\s+standard|widely\s+used|de\s+facto\s+standard)\b",
				r"\b(popular|common)\s+(choice|approach|solution)\b",
			],
			"weak": [
				r"\bwidely\s+used\b",
				r"\bindustry\s+standard\b",
				r"\bpopular\b",
				r"\bcommon\b",
				r"\bstandard\b",
			],
		},
		"overconfidence_cues": {
			"strong": [
				r"\b(no\s+doubt|definitely|certainly|absolutely)\b",
				r"\bI['’]m\s+(sure|certain|confident)\b",
				r"\bguarantee\b",
			],
			"weak": [
				r"\bdefinitely\b",
				r"\bsure\b",
				r"\bconfident\b",
				r"\bfor\s+sure\b",
			],
		},
		"availability_cues": {
			"strong": [
				r"\bI\s+(saw|read|heard)\b.*\b(Stack\s*Overflow|Reddit|Hacker\s*News|blog|tutorial|video|paper)\b",
				r"\b(as\s+I\s+recall|I\s+remember)\b",
			],
			"weak": [
				r"\bI\s+read\b",
				r"\bI\s+saw\b",
				r"\bI\s+heard\b",
				r"\btutorial\b",
				r"\bblog\b",
			],
		},
		"hindsight_cues": {
			"strong": [
				r"\b(in\s+hindsight|as\s+it\s+turned\s+out|looking\s+back)\b",
				r"\bshould\s+have\s+known\b",
			],
			"weak": [
				r"\bin\s+hindsight\b",
				r"\blooking\s+back\b",
			],
		},
		"time_preference_cues": {
			"strong": [
				r"\b(quick\s+fix|hacky\s+fix|temporary\s+fix)\b",
				r"\b(asap|urgent|deadline)\b",
				r"\b(by\s+tomorrow|by\s+today|this\s+week)\b",
			],
			"weak": [
				r"\bquick\b",
				r"\basap\b",
				r"\burgent\b",
				r"\bdeadline\b",
			],
		},
	}
	compiled = {}
	for feat, levels in cb.items():
		compiled[feat] = {
			"strong": [re.compile(p, re.I) for p in levels["strong"]],
			"weak": [re.compile(p, re.I) for p in levels["weak"]],
		}
	return compiled

CODEBOOK = compile_codebook()
FEATURES = list(CODEBOOK.keys())

# PROBE bias types -> feature family mapping
BIAS_TO_FEATURE = {
	"confirmation bias": "confirmation_cues",
	"anchoring bias": "anchoring_phrases",
	"framing effect": "framing_language",
	"bandwagon effect": "bandwagon_cues",
	"overconfidence bias": "overconfidence_cues",
	"availability bias": "availability_cues",
	"hindsight bias": "hindsight_cues",
	"hyperbolic discounting": "time_preference_cues",
}

def label_prompt(text: str):
	labels = {}
	for feat, levels in CODEBOOK.items():
		strong = any(r.search(text) for r in levels["strong"])
		weak = any(r.search(text) for r in levels["weak"])
		labels[feat] = {"strong": bool(strong), "weak": bool(weak), "any": bool(strong or weak)}
	return labels

# ------------------------- Input handling (zip OR directory) -------------------------

def ensure_dir(input_path: str, extract_to: str, label: str):
	p = pathlib.Path(input_path)
	if p.is_dir():
		return str(p)
	if p.is_file() and p.suffix.lower() == ".zip":
		os.makedirs(extract_to, exist_ok=True)
		with zipfile.ZipFile(str(p), "r") as z:
			z.extractall(extract_to)
		return extract_to
	raise ValueError(f"{label} must be a directory or a .zip file: {input_path}")

# ------------------------- Loading DevGPT -------------------------

PROMPT_KEYS = ["prompt", "Prompt", "instruction", "Instruction", "question", "Question", "query", "Query", "input", "Input", "content", "Content", "text", "Text"]

def _is_git_lfs_pointer(text_head: str):
	return text_head.startswith("version https://git-lfs.github.com/spec/v1")

def safe_json_load(path: str, max_bad_logs= 10, _state = {"bad": 0}):
	"""
	Safely load JSON; returns None if invalid / empty / Git-LFS pointer.
	Also tolerates JSONL content in a '.json' file.
	"""
	with open(path, "r", encoding="utf-8", errors="ignore") as fp:
		head = fp.read(400)
		if not head.strip():
			return None
		if _is_git_lfs_pointer(head):
			if _state["bad"] < max_bad_logs:
				print(f"[WARN] Git-LFS pointer file (not real JSON): {path}. Run `git lfs pull`.")
			_state["bad"] += 1
			return None
		fp.seek(0)
		return json.load(fp)

def extract_prompts_from_obj(obj):
	out = []
	def walk(x):
		if x is None:
			return
		if isinstance(x, str):
			return
		if isinstance(x, dict):
			for k in PROMPT_KEYS:
				if k in x and isinstance(x[k], str) and x[k].strip():
					out.append(x[k])
			for ck in ["conversations", "conversation", "messages", "turns", "dialog", "chat"]:
				if ck in x and isinstance(x[ck], list):
					for m in x[ck]:
						if isinstance(m, dict):
							role = m.get("role") or m.get("from") or m.get("speaker") or m.get("author")
							text = m.get("content") or m.get("text") or m.get("value") or m.get("message")
							if isinstance(text, str) and text.strip():
								if role:
									role_s = str(role).lower()
									if role_s in ["user", "human", "prompter", "questioner"]:
										out.append(text)
								else:
									out.append(text)
						elif isinstance(m, str) and m.strip():
							out.append(m)
			for v in x.values():
				walk(v)
		elif isinstance(x, list):
			for it in x:
				walk(it)
	walk(obj)

	# de-dup preserving order
	seen = set()
	cleaned = []
	for p in out:
		ps = p.strip()
		if not ps:
			continue
		h = hashlib.sha1(ps.encode("utf-8", "ignore")).hexdigest()
		if h in seen:
			continue
		seen.add(h)
		cleaned.append(ps)
	return cleaned

def load_devgpt_from_dir(root_dir: str, max_bad_json_logs= 10):
	"""
	Walk root_dir and extract prompts from .json/.jsonl/.csv/.tsv
	Returns:
	  prompts
	  meta: list[(rel_source_file, idx)]
	  candidates
	"""
	root_dir = str(pathlib.Path(root_dir))
	prompts = []
	meta: List[Tuple[str, Any]] = []
	candidates = []

	for p in pathlib.Path(root_dir).rglob("*"):
		if p.is_file() and p.suffix.lower() in [".json", ".jsonl", ".csv", ".tsv"] and "__MACOSX" not in str(p):
			candidates.append(str(p))
	candidates.sort()

	bad_state = {"bad": 0}

	for f in tqdm(candidates, desc="Scanning DevGPT files"):
		ext = pathlib.Path(f).suffix.lower()
		rel = os.path.relpath(f, root_dir)
		if ext == ".json":
			obj = safe_json_load(f, max_bad_logs=max_bad_json_logs, _state=bad_state)
			if obj is None:
				continue
			ps = extract_prompts_from_obj(obj)
			for i, ptxt in enumerate(ps):
				prompts.append(ptxt)
				meta.append((rel, i))
		elif ext == ".jsonl":
			with open(f, "r", encoding="utf-8", errors="ignore") as fp:
				for li, line in enumerate(fp):
					line = line.strip()
					if not line:
						continue
					try:
						obj = json.loads(line)
					except Exception:
						continue
					ps = extract_prompts_from_obj(obj)
					for i, ptxt in enumerate(ps):
						prompts.append(ptxt)
						meta.append((rel, f"{li}:{i}"))
		elif ext in [".csv", ".tsv"]:
			delim = "," if ext == ".csv" else "\t"
			with open(f, "r", encoding="utf-8", errors="ignore", newline="") as fp:
				reader = csv.DictReader(fp, delimiter=delim)
				for ri, row in enumerate(reader):
					found = None
					for k in PROMPT_KEYS:
						if k in row and row[k] and str(row[k]).strip():
							found = row[k]
							break
					if found:
						prompts.append(str(found))
						meta.append((rel, ri))

	return prompts, meta, candidates

# ------------------------- Loading PROBE-SWE + cue docs -------------------------

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def inserted_spans(biased: str, unbiased: str):
	bt = TOKEN_RE.findall(biased)
	ut = TOKEN_RE.findall(unbiased)
	sm = difflib.SequenceMatcher(a=ut, b=bt)
	spans = []
	for tag, i1, i2, j1, j2 in sm.get_opcodes():
		if tag == "insert":
			s = " ".join(bt[j1:j2]).strip()
			if s:
				spans.append(s)
		elif tag == "replace":
			seq = bt[j1:j2]
			s = " ".join(seq).strip()
			if s and len(seq) >= 2:
				spans.append(s)
	cleaned = []
	for s in spans:
		s = re.sub(r"\s+([,.;:!?])", r"\1", s)
		s = re.sub(r"\s+", " ", s).strip()
		if len(s) >= 3:
			cleaned.append(s)
	return cleaned

def find_probe_pairs(extract_dir: str):
	"""
	Supports:
	  1) JSON dict-of-biases: { "<bias>": [ { "biased":..., "unbiased":... }, ...], ... }
	  2) JSON list of items with biased/unbiased keys
	  3) CSV/TSV with biased/unbiased columns
	"""
	pairs = []
	files = []
	for p in pathlib.Path(extract_dir).rglob("*"):
		if p.is_file() and p.suffix.lower() in [".json", ".jsonl", ".csv", ".tsv"] and "__MACOSX" not in str(p):
			files.append(str(p))
	files.sort()

	def add_pair(bt: str, b: str, u: str, src: str):
		b = (b or "").strip()
		u = (u or "").strip()
		if not b or not u:
			return
		pairs.append({"bias_type": (bt or "unknown").strip(), "biased": b, "unbiased": u, "source_file": os.path.basename(src)})

	def normalize_key(s: str):
		return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

	for f in files:
		ext = pathlib.Path(f).suffix.lower()
		if ext in [".csv", ".tsv"]:
			delim = "," if ext == ".csv" else "\t"
			with open(f, "r", encoding="utf-8", errors="ignore", newline="") as fp:
				reader = csv.DictReader(fp, delimiter=delim)
				if reader.fieldnames is None:
					continue
				nmap = {fn: normalize_key(fn) for fn in reader.fieldnames}
				biased_col = unbiased_col = bias_col = None
				for fn, nk in nmap.items():
					if "biased" in nk and ("prompt" in nk or "text" in nk or nk == "biased"):
						biased_col = fn
					if ("unbiased" in nk or "neutral" in nk) and ("prompt" in nk or "text" in nk or nk == "unbiased"):
						unbiased_col = fn
					if nk in ["bias_type", "bias", "type", "category", "label"]:
						bias_col = fn
				if not (biased_col and unbiased_col):
					continue
				for row in reader:
					add_pair(str(row.get(bias_col) or "unknown"), str(row.get(biased_col) or ""), str(row.get(unbiased_col) or ""), f)

		elif ext == ".json":
			obj = safe_json_load(f, max_bad_logs=0)  # probe should be clean; keep silent
			if obj is None:
				continue

			if isinstance(obj, dict):
				for bt, lst in obj.items():
					if isinstance(lst, list):
						for it in lst:
							if isinstance(it, dict):
								if "valid" in it and it["valid"] is False:
									continue
								if "biased" in it and "unbiased" in it:
									add_pair(str(bt), str(it.get("biased") or ""), str(it.get("unbiased") or ""), f)
								else:
									keys = {k.lower(): k for k in it.keys()}
									bkey = next((ok for lk, ok in keys.items() if "biased" in lk), None)
									ukey = next((ok for lk, ok in keys.items() if "unbiased" in lk or "neutral" in lk), None)
									if bkey and ukey:
										add_pair(str(bt), str(it.get(bkey) or ""), str(it.get(ukey) or ""), f)
			elif isinstance(obj, list):
				for it in obj:
					if not isinstance(it, dict):
						continue
					keys = {k.lower(): k for k in it.keys()}
					bkey = next((ok for lk, ok in keys.items() if "biased" in lk), None)
					ukey = next((ok for lk, ok in keys.items() if "unbiased" in lk or "neutral" in lk), None)
					if bkey and ukey:
						bt = it.get("bias_type") or it.get("bias") or it.get("type") or it.get("category") or it.get("label") or "unknown"
						if "valid" in it and it["valid"] is False:
							continue
						add_pair(str(bt), str(it.get(bkey) or ""), str(it.get(ukey) or ""), f)

	return pairs, files

def build_cue_docs(pairs):
	cue_docs: Dict[str, List[str]] = defaultdict(list)
	for it in pairs:
		spans = inserted_spans(it["biased"], it["unbiased"])
		if spans:
			cue_docs[it["bias_type"] or "unknown"].extend(spans)
	return {k: " ".join(v) for k, v in cue_docs.items() if v}

# ------------------------- Similarity (TF-IDF baseline) -------------------------

def tfidf_cosine_matrix(docs, queries):
	"""
	Cosine similarity of queries -> docs using TF-IDF.
	Returns matrix shape (len(queries), len(docs)).
	"""
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity
	vec = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2))
	X = vec.fit_transform(docs)
	Q = vec.transform(queries)
	return cosine_similarity(Q, X)

# ------------------------- "AI" labeling (SBERT) -------------------------

def _try_import_sbert():
	from sentence_transformers import SentenceTransformer  # type: ignore
	return SentenceTransformer

_SBERT_CACHE = {}

def pick_device(user_device: str):
	"""
	Pick a torch device. In auto mode: prefer CUDA, then Apple MPS, else CPU.
	"""
	if user_device and user_device != "auto":
		return user_device
	import torch  # type: ignore
	if hasattr(torch, "cuda") and torch.cuda.is_available():
		return "cuda"
	# Apple Silicon / Metal (MPS)
	mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()
	if mps_ok:
		return "mps"
	return "cpu"

def sbert_encode(texts, model_name: str, device: str = "auto"):
	"""
	Encode texts using Sentence-Transformers with automatic device selection.

	Adds a persistent on-disk embedding cache to avoid recomputing embeddings across runs.
	The cache is keyed by (model_name, sha1(text)) and stores *normalized* embeddings so
	cosine similarity = dot product.

	Cache directory:
	- Set via --emb_cache_dir (preferred) or $SBERT_EMB_CACHE_DIR
	- Falls back to ./cache_embeddings
	"""
	import numpy as np

	SentenceTransformer = _try_import_sbert()
	if SentenceTransformer is None:
		raise RuntimeError("sentence-transformers not installed. Install: pip install -U sentence-transformers")

	dev = pick_device(device)
	cache_key = (model_name, dev)
	model = _SBERT_CACHE.get(cache_key)
	if model is None:
		model = SentenceTransformer(model_name, device=dev)
		_SBERT_CACHE[cache_key] = model

	# -------- persistent embedding cache --------
	cache_dir = _SBERT_EMB_CACHE_DIR or "cache_embeddings"
	os.makedirs(cache_dir, exist_ok=True)
	safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
	cache_file = os.path.join(cache_dir, f"sbert_embeddings_{safe_model}_norm.pkl")

	disk_cache = _SBERT_EMB_DISK_CACHES.get(cache_file)
	if disk_cache is None:
		disk_cache = _load_or_create_cache(cache_file, lambda: {}) or {}
		_SBERT_EMB_DISK_CACHES[cache_file] = disk_cache

	def _key_fn(t: str):
		return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()

	def _fetch_fn(missing_values):
		missing_list = list(missing_values)
		if not missing_list:
			return
		emb = model.encode(
			missing_list,
			batch_size=64,
			convert_to_numpy=True,
			normalize_embeddings=True,
			show_progress_bar=True,
		)
		for t, e in zip(missing_list, emb):
			yield t, e

	# NOTE: empty_is_missing must be False because numpy arrays have ambiguous truthiness.
	emb_list = _get_cached_values(
		texts,
		disk_cache,
		_fetch_fn,
		cache_name=cache_file,
		key_fn=_key_fn,
		empty_is_missing=False,
	)

	if any(e is None for e in emb_list):
		raise ValueError("Encountered empty text while encoding; ensure inputs are non-empty strings.")

	return np.vstack(emb_list).astype("float32", copy=False)


def cosine_sim_matrix(A, B):
	# A: (n,d) normalized, B: (m,d) normalized
	return A @ B.T

def ai_map_sbert_similarity(dev_texts, cue_docs, model_name: str, threshold: float, topk= 1, device: str = "auto"):
	"""
	Semantic mapping: prompt -> most similar PROBE cue-doc.
	"""
	import numpy as np
	bias_types = sorted(cue_docs.keys())
	cue_texts = [cue_docs[b] for b in bias_types]
	if not dev_texts or not cue_texts:
		return [{"ai_best_bias": "", "ai_best_score": 0.0, "ai_mapped": 0, "ai_top": []} for _ in dev_texts]
	dev_emb = sbert_encode(dev_texts, model_name, device=device)
	cue_emb = sbert_encode(cue_texts, model_name, device=device)
	S = cosine_sim_matrix(dev_emb, cue_emb)  # (n_prompts, n_biases)
	rows = []
	for i in range(S.shape[0]):
		sims = S[i]
		order = np.argsort(-sims)
		top = [(bias_types[j], float(sims[j])) for j in order[: max(1, topk)]]
		best_bias, best_score = top[0]
		rows.append({"ai_best_bias": best_bias, "ai_best_score": best_score, "ai_mapped": int(best_score >= threshold), "ai_top": top})
	return rows

def train_probe_sbert_classifier(
	probe_pairs,
	model_name: str,
	train_text: str = "both",
	min_samples_per_class= 8,
	class_weight: str = "balanced",
	device: str = "auto",
	eval_mode: str = "holdout",
	eval_test_size: float = 0.2,
	eval_seed: int = 1337,
	eval_threshold: Optional[float] = None,
	return_metrics: bool = False,
):
	"""
	Trains a lightweight classifier on PROBE cue language using SBERT embeddings + LogisticRegression.

	Robustness improvements:
	- Can train on inserted cue spans, full biased prompts, or both (default=both).
	- Drops classes with too few instances (min_samples_per_class).
	- Uses class_weight='balanced' by default to mitigate imbalance.
	- Deterministic random_state.

	Returns (clf, label_encoder, kept_classes) or (None, None, []).

	NOTE: This is still fully replicable: training data is PROBE only, model is named, and hyperparams are fixed.
	"""

	from sklearn.linear_model import LogisticRegression
	from sklearn.preprocessing import LabelEncoder

	X_text = []
	y = []

	for it in probe_pairs:
		bt = str(it.get("bias_type", "unknown"))
		if train_text in ("spans", "both"):
			spans = inserted_spans(it.get("biased", ""), it.get("unbiased", ""))
			for s in spans:
				if len(s) >= 3:
					X_text.append(s)
					y.append(bt)
		if train_text in ("biased", "both"):
			b = str(it.get("biased", "")).strip()
			if b:
				X_text.append(b)
				y.append(bt)

	if not X_text:
		return (None, None, [], {}) if return_metrics else (None, None, [])

	# Drop rare classes to avoid unstable training
	counts = Counter(y)
	kept = {c for c, n in counts.items() if n >= min_samples_per_class}
	if len(kept) < 2:
		# Not enough classes left to learn a classifier
		return (None, None, [], {}) if return_metrics else (None, None, [])

	X_text_f = [t for t, c in zip(X_text, y) if c in kept]
	y_f = [c for c in y if c in kept]

	X = sbert_encode(X_text_f, model_name, device=device)
	# Guard against rare NaN/inf values in embeddings (prevents sklearn linear algebra warnings)
	import numpy as np
	X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
	le = LabelEncoder()
	y_enc = le.fit_transform(y_f)

	cw = None if class_weight == "none" else "balanced"
	clf = LogisticRegression(
		max_iter=3000,
		multi_class="auto",
		class_weight=cw,
		random_state=1337,
	)
	clf.fit(X, y_enc)
	kept_classes = list(le.classes_)

	# Optional holdout evaluation (PROBE only) to estimate classifier quality
	clf_metrics = {}
	if eval_mode and str(eval_mode) != "none":
		try:
			import numpy as np
			from sklearn.model_selection import StratifiedShuffleSplit
			from sklearn.metrics import accuracy_score, precision_recall_fscore_support

			# Need enough samples per class for a stratified split
			class_counts = Counter(list(y_enc))
			min_per_class = min(class_counts.values()) if class_counts else 0
			if len(class_counts) >= 2 and min_per_class >= 2 and len(y_enc) >= 20:
				splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(eval_test_size), random_state=int(eval_seed))
				train_idx, test_idx = next(splitter.split(X, y_enc))

				clf_eval = LogisticRegression(
					max_iter=3000,
					multi_class="auto",
					class_weight=cw,
					random_state=1337,
				)
				clf_eval.fit(X[train_idx], y_enc[train_idx])
				y_pred = clf_eval.predict(X[test_idx])

				acc = float(accuracy_score(y_enc[test_idx], y_pred))
				p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
					y_enc[test_idx], y_pred, average="macro", zero_division=0
				)
				p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
					y_enc[test_idx], y_pred, average="weighted", zero_division=0
				)

				clf_metrics = {
					"eval_mode": str(eval_mode),
					"eval_test_size": float(eval_test_size),
					"eval_seed": int(eval_seed),
					"n_samples": int(len(y_enc)),
					"n_classes": int(len(class_counts)),
					"accuracy": acc,
					"precision_macro": float(p_macro),
					"recall_macro": float(r_macro),
					"f1_macro": float(f_macro),
					"precision_weighted": float(p_weighted),
					"recall_weighted": float(r_weighted),
					"f1_weighted": float(f_weighted),
				}

				# Optional abstention based on max-prob threshold (coverage + accuracy on covered subset)
				if eval_threshold is not None:
					thr = float(eval_threshold)
					proba = clf_eval.predict_proba(X[test_idx])
					maxp = proba.max(axis=1)
					mask = maxp >= thr
					coverage = float(mask.mean()) if len(mask) else 0.0
					acc_cov = ""
					if mask.any():
						y_pred_cov = np.argmax(proba[mask], axis=1)
						acc_cov = float(accuracy_score(y_enc[test_idx][mask], y_pred_cov))
					clf_metrics.update({
						"eval_threshold": thr,
						"coverage": coverage,
						"accuracy_on_covered": acc_cov,
					})
			else:
				clf_metrics = {
					"eval_mode": str(eval_mode),
					"note": "skipped_eval_insufficient_data",
					"n_samples": int(len(y_enc)),
					"n_classes": int(len(class_counts)),
					"min_per_class": int(min_per_class),
				}
		except Exception as e:
			clf_metrics = {"eval_mode": str(eval_mode), "note": f"eval_error:{type(e).__name__}"}

	if return_metrics:
		return clf, le, kept_classes, clf_metrics
	return clf, le, kept_classes

def ai_predict_probe_sbert_classifier(dev_texts, clf, le, model_name: str, threshold: float, topk= 1, device: str = "auto"):
	import numpy as np
	X = sbert_encode(dev_texts, model_name, device=device)
	X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
	proba = clf.predict_proba(X)  # (n, n_classes)
	rows = []
	for i in range(proba.shape[0]):
		p = proba[i]
		order = np.argsort(-p)
		top = [(le.inverse_transform([j])[0], float(p[j])) for j in order[: max(1, topk)]]
		best_bias, best_score = top[0]
		rows.append({"ai_best_bias": best_bias, "ai_best_score": best_score, "ai_mapped": int(best_score >= threshold), "ai_top": top})
	return rows

# ------------------------- "AI" labeling (HF finetuned classifier) -------------------------

def _default_user_cache_home():
	"""
	Return a stable per-user cache home directory.
	Honors HF_BIAS_CLF_CACHE_DIR, then XDG_CACHE_HOME, else ~/.cache.
	"""
	base = os.environ.get("HF_BIAS_CLF_CACHE_DIR", "").strip()
	if base:
		return os.path.expanduser(base)
	xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
	if xdg:
		return os.path.join(os.path.expanduser(xdg), "bias_features_devgpt")
	return os.path.join(os.path.expanduser("~"), ".cache", "bias_features_devgpt")

def _hash_probe_pairs(probe_pairs):
	"""
	Stable hash of PROBE training data used for caching.
	Order-insensitive by sorting (bias_type, biased, unbiased).
	"""
	h = hashlib.sha1()
	items = []
	for it in probe_pairs or []:
		bt = str(it.get("bias_type", ""))
		b = str(it.get("biased", ""))
		u = str(it.get("unbiased", ""))
		items.append((bt, b, u))
	items.sort()
	for bt, b, u in items:
		h.update(bt.encode("utf-8", errors="ignore"))
		h.update(b"\0")
		h.update(b.encode("utf-8", errors="ignore"))
		h.update(b"\0")
		h.update(u.encode("utf-8", errors="ignore"))
		h.update(b"\n")
	return h.hexdigest()

def _hf_training_signature(
	probe_pairs,
	model_name: str,
	train_text: str,
	min_samples_per_class: int,
	eval_mode: str,
	eval_test_size: float,
	eval_seed: int,
	max_len: int,
	lr: float,
	epochs: int,
	train_batch_size: int,
	eval_batch_size: int,
	weight_decay: float,
	warmup_ratio: float,
	grad_accum: int,
):
	probe_hash = _hash_probe_pairs(probe_pairs)
	sig = {
		"probe_hash": probe_hash,
		"model_name": str(model_name),
		"train_text": str(train_text),
		"min_samples_per_class": int(min_samples_per_class),
		"eval_mode": str(eval_mode),
		"eval_test_size": float(eval_test_size),
		"eval_seed": int(eval_seed),
		"max_len": int(max_len),
		"lr": float(lr),
		"epochs": int(epochs),
		"train_batch_size": int(train_batch_size),
		"eval_batch_size": int(eval_batch_size),
		"weight_decay": float(weight_decay),
		"warmup_ratio": float(warmup_ratio),
		"grad_accum": int(grad_accum),
	}
	key = hashlib.sha1(json.dumps(sig, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
	sig["cache_key"] = key
	return sig

def resolve_hf_out_dir(args, probe_pairs, default_out_dir: str = ""):
	"""
	Resolve the directory used to save/load the finetuned HF classifier.

	- If --hf_out_dir is provided, use it verbatim.
	- Otherwise, use a stable per-user cache directory keyed by PROBE + training params.
	  (Set HF_BIAS_CLF_CACHE_DIR to override the base cache location.)
	- If the cache directory cannot be created, fall back to default_out_dir (if provided).
	"""
	explicit = (getattr(args, "hf_out_dir", "") or "").strip()
	if explicit:
		return explicit

	sig = _hf_training_signature(
		probe_pairs,
		model_name=getattr(args, "hf_model", ""),
		train_text=getattr(args, "clf_train_text", "both"),
		min_samples_per_class=int(getattr(args, "clf_min_samples_per_class", 8)),
		eval_mode=str(getattr(args, "clf_eval", "holdout")),
		eval_test_size=float(getattr(args, "clf_eval_test_size", 0.2)),
		eval_seed=int(getattr(args, "clf_eval_seed", 1337)),
		max_len=int(getattr(args, "hf_max_len", 256)),
		lr=float(getattr(args, "hf_lr", 2e-5)),
		epochs=int(getattr(args, "hf_epochs", 3)),
		train_batch_size=int(getattr(args, "hf_train_batch_size", 16)),
		eval_batch_size=int(getattr(args, "hf_eval_batch_size", 512)),
		weight_decay=float(getattr(args, "hf_weight_decay", 0.01)),
		warmup_ratio=float(getattr(args, "hf_warmup_ratio", 0.06)),
		grad_accum=int(getattr(args, "hf_grad_accum", 1)),
	)
	key = str(sig.get("cache_key", ""))[:12]
	model_name = str(getattr(args, "hf_model", "hf")).strip() or "hf"
	safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)[:80]
	root = os.path.join(_default_user_cache_home(), "hf_bias_clf")
	auto_dir = os.path.join(root, f"{safe_model}-{key}")

	try:
		os.makedirs(auto_dir, exist_ok=True)
		return auto_dir
	except Exception:
		if default_out_dir:
			os.makedirs(default_out_dir, exist_ok=True)
			return default_out_dir
		return auto_dir

def _try_import_hf():
	try:
		import torch  # type: ignore
		from transformers import (  # type: ignore
			AutoTokenizer,
			AutoModelForSequenceClassification,
			Trainer,
			TrainingArguments,
			set_seed,
		)
		return torch, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
	except Exception as e:
		raise RuntimeError(
			"HF finetuning requires torch + transformers. Install e.g.: "
			"pip install -U torch transformers"
		) from e

class _TorchTextClsDataset:
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		item = {k: v[idx] for k, v in self.encodings.items()}
		item["labels"] = self.labels[idx]
		return item

def train_probe_hf_classifier(
	probe_pairs,
	model_name: str,
	out_dir: str,
	train_text: str = "both",
	min_samples_per_class: int = 8,
	eval_mode: str = "holdout",
	eval_test_size: float = 0.2,
	eval_seed: int = 1337,
	max_len: int = 256,
	lr: float = 2e-5,
	epochs: int = 3,
	train_batch_size: int = 16,
	eval_batch_size: int = 32,
	weight_decay: float = 0.01,
	warmup_ratio: float = 0.06,
	grad_accum: int = 1,
	fp16: bool = False,
	num_workers: int = 2,
	device: str = "auto",
	retrain: bool = False,
	eval_threshold: Optional[float] = None,
	return_metrics: bool = False,
):
	"""
	Finetune a transformer classifier on PROBE (bias_type prediction).

	Saves:
	- model + tokenizer to out_dir
	- label mapping to out_dir/label_encoder.json
	- training signature to out_dir/train_signature.json

	CHANGE vs old version:
	- If a saved model is reused (signature matches), we STILL recompute holdout metrics
	  on PROBE (unless eval_mode == "none"), so classifier_quality.csv is updated.
	"""
	import numpy as np
	from sklearn.preprocessing import LabelEncoder

	label_path = os.path.join(out_dir, "label_encoder.json")
	sig_path = os.path.join(out_dir, "train_signature.json")

	signature = _hf_training_signature(
		probe_pairs,
		model_name=model_name,
		train_text=train_text,
		min_samples_per_class=min_samples_per_class,
		eval_mode=eval_mode,
		eval_test_size=eval_test_size,
		eval_seed=eval_seed,
		max_len=max_len,
		lr=lr,
		epochs=epochs,
		train_batch_size=train_batch_size,
		eval_batch_size=eval_batch_size,
		weight_decay=weight_decay,
		warmup_ratio=warmup_ratio,
		grad_accum=grad_accum,
	)
	cache_key = str(signature.get("cache_key", ""))

	def _maybe_write_signature():
		try:
			os.makedirs(out_dir, exist_ok=True)
			with open(sig_path, "w", encoding="utf-8") as fp:
				json.dump(signature, fp, ensure_ascii=False, indent=2, sort_keys=True)
		except Exception:
			pass

	def _eval_reused_model_on_probe(model, tokenizer, le):
		"""
		Recompute PROBE holdout quality stats even when reusing a saved model.
		Returns a clf_metrics dict compatible with the training path metrics.
		"""
		from sklearn.model_selection import StratifiedShuffleSplit
		from sklearn.metrics import accuracy_score, precision_recall_fscore_support

		n_labels = int(len(getattr(le, "classes_", [])))

		# Eval disabled explicitly
		if not eval_mode or str(eval_mode) == "none":
			return {
				"eval_mode": str(eval_mode),
				"note": "eval_disabled",
				"n_samples": 0,
				"n_classes": n_labels,
				"model_source": "reuse",
			}

		# Rebuild eval dataset exactly like training path (but restricted to kept label set)
		X_text = []
		y = []
		kept_set = set([str(c) for c in getattr(le, "classes_", [])])

		for it in probe_pairs:
			bt = str(it.get("bias_type", "unknown"))
			if bt not in kept_set:
				continue

			if train_text in ("spans", "both"):
				spans = inserted_spans(it.get("biased", ""), it.get("unbiased", ""))
				for s in spans:
					s = strip_code(str(s))
					if len(s) >= 3:
						X_text.append(s)
						y.append(bt)

			if train_text in ("biased", "both"):
				b = strip_code(str(it.get("biased", "")).strip())
				if b:
					X_text.append(b)
					y.append(bt)

		if not X_text:
			return {
				"eval_mode": str(eval_mode),
				"note": "no_data_for_eval",
				"n_samples": 0,
				"n_classes": n_labels,
				"model_source": "reuse",
			}

		# Encode labels using the loaded label encoder
		try:
			y_enc = le.transform(y).astype("int64")
		except Exception:
			return {
				"eval_mode": str(eval_mode),
				"note": "label_encode_failed",
				"n_samples": int(len(y)),
				"n_classes": n_labels,
				"model_source": "reuse",
			}

		n_samples = int(len(y_enc))

		# Stratified holdout split
		try:
			splitter = StratifiedShuffleSplit(
				n_splits=1,
				test_size=float(eval_test_size),
				random_state=int(eval_seed),
			)
			_, eval_idx = next(splitter.split(np.zeros_like(y_enc), y_enc))
		except Exception:
			eval_idx = np.array([], dtype=int)

		if len(eval_idx) == 0:
			return {
				"eval_mode": str(eval_mode),
				"note": "no_eval_split",
				"n_samples": n_samples,
				"n_classes": n_labels,
				"model_source": "reuse",
			}

		# Batched inference on eval split
		torch, *_ = _try_import_hf()
		dev = pick_device(device)

		model.eval()
		model.to(dev)
		if bool(fp16) and dev == "cuda":
			try:
				model.half()
			except Exception:
				pass

		X_eval = [X_text[i] for i in eval_idx]
		y_eval = y_enc[eval_idx]

		all_logits = []
		bs = max(1, int(eval_batch_size))

		for start in range(0, len(X_eval), bs):
			batch = X_eval[start : start + bs]
			enc = tokenizer(
				batch,
				truncation=True,
				padding=True,
				max_length=int(max_len),
				return_tensors="pt",
			)
			enc = {k: v.to(dev) for k, v in enc.items()}
			with torch.no_grad():
				out = model(**enc)
				all_logits.append(out.logits.detach().cpu().numpy())

		if not all_logits:
			return {
				"eval_mode": str(eval_mode),
				"note": "no_logits",
				"n_samples": n_samples,
				"n_classes": n_labels,
				"model_source": "reuse",
			}

		logits = np.vstack(all_logits)
		pred = np.argmax(logits, axis=1)

		acc = float(accuracy_score(y_eval, pred))
		p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
			y_eval, pred, average="macro", zero_division=0
		)
		p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
			y_eval, pred, average="weighted", zero_division=0
		)

		clf_metrics = {
			"eval_mode": str(eval_mode),
			"eval_test_size": float(eval_test_size),
			"eval_seed": int(eval_seed),
			"n_samples": n_samples,
			"n_classes": n_labels,
			"accuracy": acc,
			"precision_macro": float(p_macro),
			"recall_macro": float(r_macro),
			"f1_macro": float(f_macro),
			"precision_weighted": float(p_weighted),
			"recall_weighted": float(r_weighted),
			"f1_weighted": float(f_weighted),
			"model_source": "reuse",
			# IMPORTANT: keep note empty so your main() print branch triggers
			"note": "",
		}

		# Optional abstention stats
		if eval_threshold is not None:
			thr = float(eval_threshold)
			proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
			maxp = proba.max(axis=1)
			mask = maxp >= thr
			coverage = float(mask.mean()) if len(mask) else 0.0
			acc_cov = ""
			if mask.any():
				acc_cov = float(accuracy_score(y_eval[mask], np.argmax(proba[mask], axis=1)))
			clf_metrics.update({
				"eval_threshold": thr,
				"coverage": coverage,
				"accuracy_on_covered": acc_cov,
			})

		return clf_metrics

	# ----------------------------
	# Reuse if already trained and signature matches (or signature missing; backward compatible)
	# ----------------------------
	if (not retrain) and os.path.isdir(out_dir) and os.path.isfile(label_path):
		sig_ok = True
		if os.path.isfile(sig_path):
			try:
				with open(sig_path, "r", encoding="utf-8") as fp:
					old = json.load(fp)
				sig_ok = str(old.get("cache_key", "")) == cache_key
			except Exception:
				sig_ok = True

		if sig_ok:
			try:
				torch, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed = _try_import_hf()
				model = AutoModelForSequenceClassification.from_pretrained(out_dir)
				tokenizer = AutoTokenizer.from_pretrained(out_dir, use_fast=True)
				with open(label_path, "r", encoding="utf-8") as fp:
					classes = json.load(fp)
				le = LabelEncoder()
				le.fit(list(classes))
				kept_classes = list(le.classes_)
				_maybe_write_signature()

				if return_metrics:
					clf_metrics = _eval_reused_model_on_probe(model, tokenizer, le)
					return model, tokenizer, le, kept_classes, clf_metrics
				return model, tokenizer, le, kept_classes
			except Exception:
				pass

	# ----------------------------
	# Build training texts
	# ----------------------------
	X_text = []
	y = []
	for it in probe_pairs:
		bt = str(it.get("bias_type", "unknown"))
		if train_text in ("spans", "both"):
			spans = inserted_spans(it.get("biased", ""), it.get("unbiased", ""))
			for s in spans:
				s = strip_code(str(s))
				if len(s) >= 3:
					X_text.append(s)
					y.append(bt)
		if train_text in ("biased", "both"):
			b = strip_code(str(it.get("biased", "")).strip())
			if b:
				X_text.append(b)
				y.append(bt)

	if not X_text:
		return (None, None, None, [], {}) if return_metrics else (None, None, None, [])

	# Drop rare classes for stability
	counts = Counter(y)
	kept = {c for c, n in counts.items() if n >= int(min_samples_per_class)}
	if len(kept) < 2:
		return (None, None, None, [], {"note": "insufficient_classes"}) if return_metrics else (None, None, None, [])

	X_text_f = [t for t, c in zip(X_text, y) if c in kept]
	y_f = [c for c in y if c in kept]

	le = LabelEncoder()
	y_enc = le.fit_transform(y_f).astype("int64")
	kept_classes = list(le.classes_)

	# Split (holdout)
	train_idx = np.arange(len(y_enc))
	eval_idx = np.array([], dtype=int)
	if eval_mode and str(eval_mode) != "none":
		try:
			from sklearn.model_selection import StratifiedShuffleSplit
			splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(eval_test_size), random_state=int(eval_seed))
			train_idx, eval_idx = next(splitter.split(np.zeros_like(y_enc), y_enc))
		except Exception:
			train_idx = np.arange(len(y_enc))
			eval_idx = np.array([], dtype=int)

	torch, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed = _try_import_hf()
	set_seed(int(eval_seed))

	os.makedirs(out_dir, exist_ok=True)
	with open(label_path, "w", encoding="utf-8") as fp:
		json.dump(list(kept_classes), fp, ensure_ascii=False, indent=2)

	dev = pick_device(device)
	use_fp16 = bool(fp16 and dev == "cuda")

	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(kept_classes))

	# Tokenize
	X_train = [X_text_f[i] for i in train_idx]
	y_train = [int(y_enc[i]) for i in train_idx]
	train_enc = tokenizer(
		X_train,
		truncation=True,
		padding=True,
		max_length=int(max_len),
		return_tensors="pt",
	)
	train_ds = _TorchTextClsDataset(train_enc, torch.tensor(y_train, dtype=torch.long))

	eval_ds = None
	if len(eval_idx) > 0:
		X_eval = [X_text_f[i] for i in eval_idx]
		y_eval = [int(y_enc[i]) for i in eval_idx]
		eval_enc = tokenizer(
			X_eval,
			truncation=True,
			padding=True,
			max_length=int(max_len),
			return_tensors="pt",
		)
		eval_ds = _TorchTextClsDataset(eval_enc, torch.tensor(y_eval, dtype=torch.long))

	def compute_metrics(eval_pred):
		from sklearn.metrics import accuracy_score, precision_recall_fscore_support
		logits, labels = eval_pred
		pred = np.argmax(logits, axis=1)
		acc = float(accuracy_score(labels, pred))
		p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
		p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(labels, pred, average="weighted", zero_division=0)
		return {
			"accuracy": acc,
			"precision_macro": float(p_macro),
			"recall_macro": float(r_macro),
			"f1_macro": float(f_macro),
			"precision_weighted": float(p_weighted),
			"recall_weighted": float(r_weighted),
			"f1_weighted": float(f_weighted),
		}

	eval_strategy = "epoch" if eval_ds is not None else "no"
	save_strategy = "epoch" if eval_ds is not None else "no"

	train_args = TrainingArguments(
		output_dir=out_dir,
		overwrite_output_dir=True,
		num_train_epochs=float(epochs),
		learning_rate=float(lr),
		per_device_train_batch_size=int(train_batch_size),
		per_device_eval_batch_size=int(eval_batch_size),
		gradient_accumulation_steps=max(1, int(grad_accum)),
		weight_decay=float(weight_decay),
		warmup_ratio=float(warmup_ratio),
		logging_steps=50,
		eval_strategy=eval_strategy,
		save_strategy=save_strategy,
		save_total_limit=1,
		load_best_model_at_end=bool(eval_ds is not None),
		metric_for_best_model="f1_macro",
		greater_is_better=True,
		dataloader_num_workers=int(num_workers) if int(num_workers) > 0 else 0,
		fp16=use_fp16,
		report_to=[],  # disable wandb/etc by default
	)

	trainer = Trainer(
		model=model,
		args=train_args,
		train_dataset=train_ds,
		eval_dataset=eval_ds,
		compute_metrics=compute_metrics if eval_ds is not None else None,
	)

	trainer.train()

	# Save
	trainer.save_model(out_dir)
	tokenizer.save_pretrained(out_dir)

	_maybe_write_signature()

	clf_metrics = {}
	if eval_ds is not None:
		eval_out = trainer.predict(eval_ds)
		logits = eval_out.predictions
		labels = eval_out.label_ids
		m = compute_metrics((logits, labels))
		clf_metrics = {
			"eval_mode": str(eval_mode),
			"eval_test_size": float(eval_test_size),
			"eval_seed": int(eval_seed),
			"n_samples": int(len(y_enc)),
			"n_classes": int(len(kept_classes)),
			**m,
		}
		if eval_threshold is not None:
			thr = float(eval_threshold)
			proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
			maxp = proba.max(axis=1)
			mask = maxp >= thr
			coverage = float(mask.mean()) if len(mask) else 0.0
			acc_cov = ""
			if mask.any():
				from sklearn.metrics import accuracy_score
				pred_cov = np.argmax(proba[mask], axis=1)
				acc_cov = float(accuracy_score(labels[mask], pred_cov))
			clf_metrics.update({
				"eval_threshold": thr,
				"coverage": coverage,
				"accuracy_on_covered": acc_cov,
			})
	else:
		clf_metrics = {
			"eval_mode": str(eval_mode),
			"note": "no_eval_split",
			"n_samples": int(len(y_enc)),
			"n_classes": int(len(kept_classes)),
		}

	if return_metrics:
		return model, tokenizer, le, kept_classes, clf_metrics
	return model, tokenizer, le, kept_classes


def ai_predict_probe_hf_classifier(
	dev_texts,
	model,
	tokenizer,
	le,
	threshold: float,
	topk: int = 1,
	device: str = "auto",
	batch_size: int = 32,
	max_len: int = 256,
	fp16: bool = False,
):
	"""Run the finetuned HF classifier and return the same schema as other ai_* methods.

	Adds a persistent on-disk cache keyed by (model_id, label-set, max_len, sha1(text)) so repeated
	runs don't have to re-run inference for the same inputs.
	"""
	import numpy as np
	torch, *_ = _try_import_hf()

	if not dev_texts:
		return []

	dev = pick_device(device)

	# -------- persistent on-disk prediction cache (best-effort) --------
	cache_file = ""
	disk_cache = None
	try:
		cache_dir = _HF_PRED_CACHE_DIR.strip() if _HF_PRED_CACHE_DIR else ""
		if not cache_dir:
			# Fallback to a stable per-user cache location
			cache_dir = os.path.join(_default_user_cache_home(), "hf_pred_cache")
		os.makedirs(cache_dir, exist_ok=True)

		model_id = getattr(model, "name_or_path", "") or getattr(getattr(model, "config", None), "_name_or_path", "") or "hf_model"
		safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_id))[:120] or "hf_model"
		label_sig = hashlib.sha1(
			json.dumps([str(x) for x in getattr(le, "classes_", [])], ensure_ascii=False).encode("utf-8")
		).hexdigest()[:12]
		cache_file = os.path.join(cache_dir, f"hf_pred_{safe_model}_{label_sig}_maxlen{int(max_len)}.pkl")

		disk_cache = _HF_PRED_DISK_CACHES.get(cache_file)
		if disk_cache is None:
			disk_cache = _load_or_create_cache(cache_file, lambda: {}) or {}
			_HF_PRED_DISK_CACHES[cache_file] = disk_cache
	except Exception:
		cache_file = ""
		disk_cache = None

	def _key_fn(t: str):
		return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()

	def _run_inference(text_list):
		# Configure model once per inference run.
		model.eval()
		model.to(dev)
		if bool(fp16) and dev == "cuda":
			try:
				model.half()
			except Exception:
				pass

		for start in tqdm(
			range(0, len(text_list), int(batch_size)),
			total=max(1, math.ceil(len(text_list) / max(1, int(batch_size)))),
			desc="Processed data",
		):
			batch = text_list[start:start + int(batch_size)]
			enc = tokenizer(
				batch,
				truncation=True,
				padding=True,
				max_length=int(max_len),
				return_tensors="pt",
			)
			enc = {k: v.to(dev) for k, v in enc.items()}
			with torch.no_grad():
				out = model(**enc)
				logits = out.logits
				proba = torch.softmax(logits, dim=-1).detach().cpu().numpy()

			for t, p in zip(batch, proba):
				yield t, [float(x) for x in p]

	# Use the cache when possible; fall back to direct inference if cache isn't available.
	if isinstance(disk_cache, dict) and cache_file:
		proba_list = _get_cached_values(
			dev_texts,
			disk_cache,
			lambda missing: _run_inference(list(missing)),
			cache_name=cache_file,
			key_fn=_key_fn,
			empty_is_missing=False,
		)
	else:
		# No cache -> run inference on all texts.
		proba_list = [v for _, v in _run_inference(list(dev_texts))]

	labels = [str(x) for x in getattr(le, "classes_", [])]
	rows = []
	for p in proba_list:
		if p is None:
			rows.append({"ai_best_bias": "", "ai_best_score": 0.0, "ai_mapped": 0, "ai_top": []})
			continue
		p_arr = np.asarray(p, dtype="float32")
		order = np.argsort(-p_arr)
		top = [(labels[int(j)] if int(j) < len(labels) else str(int(j)), float(p_arr[int(j)])) for j in order[: max(1, int(topk))]]
		best_bias, best_score = top[0]
		rows.append({
			"ai_best_bias": best_bias,
			"ai_best_score": best_score,
			"ai_mapped": int(best_score >= float(threshold)),
			"ai_top": top,
		})

	return rows

# ------------------------- I/O helpers -------------------------

def write_csv(path: str, rows, fieldnames):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8", newline="") as fp:
		w = csv.DictWriter(fp, fieldnames=fieldnames)
		w.writeheader()
		for r in rows:
			w.writerow({k: r.get(k, "") for k in fieldnames})

def write_jsonl(path: str, rows):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as fp:
		for r in rows:
			fp.write(json.dumps(r, ensure_ascii=False) + "\n")

def to_bin(x):
	"""
	Parse human annotation values to 0/1/None.
	Accepts: 0/1, true/false, yes/no, y/n.
	"""
	if x is None:
		return None
	if isinstance(x, (int, float)):
		if math.isnan(x):
			return None
		if int(x) in (0, 1):
			return int(x)
	s = str(x).strip().lower()
	if s == "":
		return None
	if s in ("1", "true", "t", "yes", "y"):
		return 1
	if s in ("0", "false", "f", "no", "n"):
		return 0
	return None

# ------------------------- Manual validation workflow -------------------------

def make_manual_eval_set(labeled_rows):
	"""
	Manual evaluation set: all DevGPT entries that pass the AI mapping threshold
	(ai_mapped == 1). This is intentionally NOT sampled.
	"""
	return [r for r in labeled_rows if int(r.get("ai_mapped", 0)) == 1]

def write_manual_validation_files(out_dir: str, sample_rows, include_predictions: bool, fname: str):
	"""
	Creates an annotation sheet with empty annotator columns.
	"""
	out_path = os.path.join(out_dir, fname)

	base_cols = ["id", "source_file", "source_index", "prompt_clean"]
	pred_cols = []
	if include_predictions:
		pred_cols = ["probe_best_bias", "probe_best_sim", "ai_best_bias", "ai_best_score", "ai_mapped"] + [f"{feat}_any" for feat in FEATURES]

	ann_cols = []
	# for feat in FEATURES:
	# 	ann_cols.append(f"ann1_{feat}")
	# ann_cols.append("ann1_notes")
	# for feat in FEATURES:
	# 	ann_cols.append(f"ann2_{feat}")
	# ann_cols.append("ann2_notes")

	fieldnames = base_cols + pred_cols + ann_cols
	rows_out = []
	for r in sample_rows:
		row = {k: r.get(k, "") for k in base_cols}
		if include_predictions:
			row.update({k: r.get(k, "") for k in pred_cols})
		# empty annotation cells
		for c in ann_cols:
			row[c] = ""
		rows_out.append(row)

	write_csv(out_path, rows_out, fieldnames)
	return out_path

def evaluate_manual_annotations(out_dir: str, annotated_csv: str, labeled_csv: str, threshold_note: str = ""):
	"""
	Reads annotated CSV and compares to model predictions from labeled_csv.
	Produces:
	  manual_validation_metrics.csv
	  manual_validation_report.txt
	"""
	_set_csv_field_size_limit()

	# Resolve CSV paths: if a relative name was provided, prefer files inside out_dir
	def _resolve(p: str) -> str:
		if not p:
			return p
		# if p exists as-is or is absolute, keep it
		if os.path.isabs(p) and os.path.exists(p):
			return p
		if os.path.exists(p):
			return p
		cand = os.path.join(out_dir, p)
		if os.path.exists(cand):
			return cand
		return p

	labeled_csv = _resolve(labeled_csv)
	annotated_csv = _resolve(annotated_csv)

	# Load labeled predictions
	labeled = {}
	with open(labeled_csv, "r", encoding="utf-8", errors="ignore") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			labeled[str(row["id"])] = row

	# Load annotated
	ann_rows = []
	with open(annotated_csv, "r", encoding="utf-8", errors="ignore") as fp:
		reader = csv.DictReader(fp)
		for row in reader:
			ann_rows.append(row)

	# Build metrics
	metrics = []
	report_lines = []
	report_lines.append(f"Manual evaluation report ({os.path.basename(annotated_csv)})")
	if threshold_note:
		report_lines.append(threshold_note)
	report_lines.append("")

	# Optional sklearn for kappa
	from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
	have_sklearn = True
	
	# Helper: get model feature from AI label
	def model_ai_for_feature(lrow, feat: str) -> int:
		ai_mapped = int(str(lrow.get("ai_mapped", "0")) == "1")
		if not ai_mapped:
			return 0
		ai_bias = lrow.get("ai_best_bias", "")
		ai_feat = BIAS_TO_FEATURE.get(ai_bias, None)
		return 1 if (ai_feat == feat) else 0

	# Compute per-feature
	for feat in FEATURES + ["ANY_AI_MAPPED"]:
		y1 = []
		y2 = []
		y_model = []
		n_used = 0

		for r in ann_rows:
			rid = str(r.get("id", "")).strip()
			if rid == "" or rid not in labeled:
				continue
			lrow = labeled[rid]

			if feat == "ANY_AI_MAPPED":
				# any bias cue: annotator any = OR over feature columns
				a1_vals = [to_bin(r.get(f"ann1_{f}")) for f in FEATURES]
				a2_vals = [to_bin(r.get(f"ann2_{f}")) for f in FEATURES]
				a1 = 1 if any(v == 1 for v in a1_vals if v is not None) else (0 if any(v == 0 for v in a1_vals if v is not None) else None)
				a2 = 1 if any(v == 1 for v in a2_vals if v is not None) else (0 if any(v == 0 for v in a2_vals if v is not None) else None)
				m = int(str(lrow.get("ai_mapped", "0")) == "1")
			else:
				a1 = to_bin(r.get(f"ann1_{feat}"))
				a2 = to_bin(r.get(f"ann2_{feat}"))
				m = model_ai_for_feature(lrow, feat)

			# Use rows where annotator 1 exists; annotator 2 optional
			if a1 is None and a2 is None:
				continue
			n_used += 1
			if a1 is not None:
				y1.append(a1)
				y_model.append(m)

			if a2 is not None:
				y2.append(a2)

		# kappa if both annotators have enough overlap
		kappa = ""
		if have_sklearn and len(y1) > 0 and len(y2) > 0:
			# compute overlap only where both present
			y1o, y2o = [], []
			for r in ann_rows:
				rid = str(r.get("id", "")).strip()
				if rid == "" or rid not in labeled:
					continue
				if feat == "ANY_AI_MAPPED":
					a1_vals = [to_bin(r.get(f"ann1_{f}")) for f in FEATURES]
					a2_vals = [to_bin(r.get(f"ann2_{f}")) for f in FEATURES]
					a1 = 1 if any(v == 1 for v in a1_vals if v is not None) else (0 if any(v == 0 for v in a1_vals if v is not None) else None)
					a2 = 1 if any(v == 1 for v in a2_vals if v is not None) else (0 if any(v == 0 for v in a2_vals if v is not None) else None)
				else:
					a1 = to_bin(r.get(f"ann1_{feat}"))
					a2 = to_bin(r.get(f"ann2_{feat}"))
				if a1 is None or a2 is None:
					continue
				y1o.append(a1)
				y2o.append(a2)
			if len(y1o) >= 10:
				kappa = float(cohen_kappa_score(y1o, y2o))

		# model vs ann1 (precision/recall/f1)
		prec = rec = f1 = support_pos = ""
		if have_sklearn and len(y1) >= 10:
			p, r_, f, s = precision_recall_fscore_support(y1, y_model, average="binary", zero_division=0)
			prec, rec, f1, support_pos = float(p), float(r_), float(f), int(sum(y1))

		# prevalence (ann1, ann2)
		prev1 = (sum(y1) / len(y1)) if y1 else ""
		prev2 = (sum(y2) / len(y2)) if y2 else ""

		metrics.append({
			"feature": feat,
			"n_annotated_used": n_used,
			"ann1_prevalence": prev1,
			"ann2_prevalence": prev2,
			"kappa_ann1_ann2": kappa,
			"model_vs_ann1_precision": prec,
			"model_vs_ann1_recall": rec,
			"model_vs_ann1_f1": f1,
			"ann1_positive_support": support_pos,
		})

	# Write outputs
	metrics_csv = os.path.join(out_dir, "manual_validation_metrics.csv")
	write_csv(metrics_csv, metrics, list(metrics[0].keys()) if metrics else ["feature"])

	rep_path = os.path.join(out_dir, "manual_validation_report.txt")
	report_lines.append("Per-feature metrics saved to manual_validation_metrics.csv")
	report_lines.append("Notes: model comparison uses AI-mapped-to-feature (via BIAS_TO_FEATURE).")
	with open(rep_path, "w", encoding="utf-8") as fp:
		fp.write("\n".join(report_lines))

	return metrics_csv, rep_path

# ------------------------- Main pipeline -------------------------

def _ai_model_name_for_outputs(args):
	"""Return the correct model name for logs/CSV depending on ai_method."""
	return args.hf_model if getattr(args, "ai_method", "") == "hf_clf" else args.ai_model

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--probe", "--probe_zip", dest="probe_path", required=True, help="PROBE-SWE .zip OR directory")
	ap.add_argument("--devgpt", "--devgpt_zip", dest="devgpt_path", required=True, help="DevGPT .zip OR directory")
	ap.add_argument("--out_dir", required=True)

	ap.add_argument("--subset", choices=["all", "decision"], default="all")
	ap.add_argument("--exclude_source_regex", default="", help="Regex on source_file (relative path) to exclude prompts, e.g. 'hn_sharings|hn_'")

	# TF-IDF validity mapping (baseline)
	ap.add_argument("--tfidf_threshold", type=float, default=0.07, help="Threshold for TF-IDF mapping to PROBE cue-docs")

	# AI labeling
	ap.add_argument("--ai_method", choices=["none", "sbert_sim", "sbert_clf", "hf_clf"], default="hf_clf")
	ap.add_argument("--ai_model", default="sentence-transformers/all-mpnet-base-v2")
	# HuggingFace finetuned classifier (smarter than embedding+LR). Only used when --ai_method hf_clf.
	ap.add_argument("--hf_model", default="microsoft/deberta-v3-base", help="HF base model to finetune (sequence classification)")
	ap.add_argument("--hf_out_dir", default="", help="Where to save/load the finetuned HF classifier. If empty, uses a deterministic cache dir under ~/.cache/bias_features_devgpt/hf_bias_clf (keyed by PROBE + training params).")
	ap.add_argument("--hf_retrain", action="store_true", help="Force retraining even if a saved HF classifier exists in --hf_out_dir")
	ap.add_argument("--hf_epochs", type=int, default=3)
	ap.add_argument("--hf_lr", type=float, default=2e-5)
	ap.add_argument("--hf_train_batch_size", type=int, default=16)
	ap.add_argument("--hf_eval_batch_size", type=int, default=32)
	ap.add_argument("--hf_max_len", type=int, default=256)
	ap.add_argument("--hf_weight_decay", type=float, default=0.01)
	ap.add_argument("--hf_warmup_ratio", type=float, default=0.06)
	ap.add_argument("--hf_grad_accum", type=int, default=1, help="Gradient accumulation steps (useful on small GPUs)")
	ap.add_argument("--hf_fp16", action="store_true", help="Use fp16 (CUDA only) when finetuning/inferencing hf_clf")
	ap.add_argument("--hf_num_workers", type=int, default=2)

	ap.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto",
					help="Compute device for SBERT (auto prefers cuda, then mps, else cpu)")
	ap.add_argument("--ai_threshold", type=float, default=0.6, help="Threshold for AI mapping (cosine sim or class prob)")
	ap.add_argument("--ai_topk", type=int, default=1)

	# Embedding cache
	ap.add_argument("--emb_cache_dir", default="", help="Directory to cache SBERT embeddings (pickle). Default: <out_dir>/_emb_cache")

	# Robust classifier options
	ap.add_argument("--clf_train_text", choices=["spans", "biased", "both"], default="both",
					help="Training signal for sbert_clf: inserted spans, full biased prompts, or both")
	ap.add_argument("--clf_min_samples_per_class", type=int, default=8,
					help="Drop PROBE bias classes with fewer than this many training instances (improves stability)")
	ap.add_argument("--clf_class_weight", choices=["balanced", "none"], default="balanced")

	# Classifier quality evaluation (PROBE holdout)
	ap.add_argument("--clf_eval", choices=["none", "holdout"], default="holdout",
					help="Compute holdout classifier quality stats (precision/recall/F1) on PROBE for sbert_clf")
	ap.add_argument("--clf_eval_test_size", type=float, default=0.2, help="Holdout fraction for classifier evaluation")
	ap.add_argument("--clf_eval_seed", type=int, default=1337, help="Random seed for classifier holdout split")

	# ap.add_argument("--max_examples_per_bias", type=int, default=10)

	# Manual validation
	ap.add_argument("--manual_mode", choices=["none", "create", "evaluate"], default="none")
	ap.add_argument("--manual_annotated_csv", default="", help="For manual_mode=evaluate: path to filled blind CSV")
	ap.add_argument("--max_bad_json_logs", type=int, default=10, help="Max warnings about bad JSON / Git-LFS pointers")

	args = ap.parse_args()
	os.makedirs(args.out_dir, exist_ok=True)

	# Configure persistent embedding cache directory
	global _SBERT_EMB_CACHE_DIR
	_SBERT_EMB_CACHE_DIR = args.emb_cache_dir.strip() if args.emb_cache_dir else os.path.join(args.out_dir, "_emb_cache")
	os.makedirs(_SBERT_EMB_CACHE_DIR, exist_ok=True)

	# Configure persistent HF "Processed data" prediction cache directory
	global _HF_PRED_CACHE_DIR
	_HF_PRED_CACHE_DIR = os.path.join(args.out_dir, "_hf_pred_cache")
	os.makedirs(_HF_PRED_CACHE_DIR, exist_ok=True)

	probe_dir = ensure_dir(args.probe_path, os.path.join(args.out_dir, "_probe_extracted"), "PROBE-SWE")
	dev_dir = ensure_dir(args.devgpt_path, os.path.join(args.out_dir, "_devgpt_extracted"), "DevGPT")

	# Load DevGPT prompts
	prompts, meta, _ = load_devgpt_from_dir(dev_dir, max_bad_json_logs=args.max_bad_json_logs)

	# Optional filtering by source file name
	if args.exclude_source_regex:
		rx = re.compile(args.exclude_source_regex, re.I)
		kept = [(p, m) for p, m in zip(prompts, meta) if not rx.search(m[0])]
		if kept:
			prompts, meta = map(list, zip(*kept))
		else:
			prompts, meta = [], []

	# Preprocess prompts
	cleaned = [strip_code(p) for p in prompts]
	keep = [(p, c, meta[i]) for i, (p, c) in enumerate(zip(prompts, cleaned)) if c]
	prompts, cleaned, meta = zip(*keep) if keep else ([], [], [])
	prompts, cleaned, meta = list(prompts), list(cleaned), list(meta)

	if args.subset == "decision":
		idxs = [i for i, c in enumerate(cleaned) if is_decision_prompt(c)]
		prompts = [prompts[i] for i in idxs]
		cleaned = [cleaned[i] for i in idxs]
		meta = [meta[i] for i in idxs]

	# ------------------ de-duplicate by prompt_clean ------------------
	seen_clean = set()
	dedup_prompts = []
	dedup_cleaned = []
	dedup_meta = []
	dup_count = 0

	for p, c, m in zip(prompts, cleaned, meta):
		# c is already strip_code()+WS-normalized, so exact match is usually what you want
		if c in seen_clean:
			dup_count += 1
			continue
		seen_clean.add(c)
		dedup_prompts.append(p)
		dedup_cleaned.append(c)
		dedup_meta.append(m)

	if dup_count:
		print(f"[INFO] Dropped {dup_count} duplicate prompt_clean rows; kept {len(dedup_cleaned)} unique prompts.")

	prompts, cleaned, meta = dedup_prompts, dedup_cleaned, dedup_meta
	# ----------------------------------------------------------------------

	# Save extracted prompts
	dev_rows = []
	for i, (p, c, (sf, idx)) in enumerate(zip(prompts, cleaned, meta)):
		dev_rows.append({"id": i, "source_file": sf, "source_index": idx, "prompt_raw": p, "prompt_clean": c})
	write_csv(os.path.join(args.out_dir, "devgpt_prompts.csv"), dev_rows, ["id", "source_file", "source_index", "prompt_raw", "prompt_clean"])

	# Codebook labeling
	labeled_rows = []
	feat_counts = {feat: Counter() for feat in FEATURES}
	any_strong = 0
	any_any = 0

	for r in tqdm(dev_rows, desc="Codebook labeling"):
		labs = label_prompt(r["prompt_clean"])
		row = dict(r)
		row["decision_like"] = int(is_decision_prompt(r["prompt_clean"]))

		strong_any = False
		any_any_feat = False
		for feat, lv in labs.items():
			row[f"{feat}_strong"] = int(lv["strong"])
			row[f"{feat}_weak"] = int(lv["weak"])
			row[f"{feat}_any"] = int(lv["any"])
			feat_counts[feat]["strong"] += int(lv["strong"])
			feat_counts[feat]["weak"] += int(lv["weak"])
			feat_counts[feat]["any"] += int(lv["any"])
			strong_any = strong_any or lv["strong"]
			any_any_feat = any_any_feat or lv["any"]

		any_strong += int(strong_any)
		any_any += int(any_any_feat)
		labeled_rows.append(row)

	n = len(dev_rows)

	# Prevalence: codebook
	prev_rows = []
	for feat in FEATURES:
		c = feat_counts[feat]
		prev_rows.append({
			"feature": feat,
			"n_prompts": n,
			"strong_count": c["strong"],
			"strong_pct": (100.0 * c["strong"] / n) if n else 0.0,
			"weak_count": c["weak"],
			"weak_pct": (100.0 * c["weak"] / n) if n else 0.0,
			"any_count": c["any"],
			"any_pct": (100.0 * c["any"] / n) if n else 0.0,
		})
	prev_rows.append({
		"feature": "ANY_FEATURE",
		"n_prompts": n,
		"strong_count": any_strong,
		"strong_pct": (100.0 * any_strong / n) if n else 0.0,
		"weak_count": "",
		"weak_pct": "",
		"any_count": any_any,
		"any_pct": (100.0 * any_any / n) if n else 0.0,
	})
	write_csv(os.path.join(args.out_dir, "prevalence_codebook.csv"), prev_rows,
			  ["feature", "n_prompts", "strong_count", "strong_pct", "weak_count", "weak_pct", "any_count", "any_pct"])

	# PROBE cue docs
	probe_pairs, _ = find_probe_pairs(probe_dir)
	cue_docs = build_cue_docs(probe_pairs)
	write_csv(os.path.join(args.out_dir, "probe_cue_docs.csv"),
			  [{"probe_bias_type": k, "cue_doc": v} for k, v in cue_docs.items()],
			  ["probe_bias_type", "cue_doc"])

	# TF-IDF mapping to cue docs (benchmark validity baseline)
	bias_types = sorted(cue_docs.keys())
	cue_texts = [cue_docs[b] for b in bias_types]
	tfidf_prev = []
	mapped_counts = Counter()
	mapped_any = 0

	if cue_texts and cleaned:
		S = tfidf_cosine_matrix(cue_texts, cleaned)  # queries x docs; here docs=cue_texts
		for i in tqdm(range(len(cleaned)), desc="TF-IDF mapping"):
			sims = S[i, :]
			best_j = int(sims.argmax()) if len(bias_types) else -1
			best_bias = bias_types[best_j] if best_j >= 0 else ""
			best_sim = float(sims[best_j]) if best_j >= 0 else 0.0
			labeled_rows[i]["probe_best_bias"] = best_bias
			labeled_rows[i]["probe_best_sim"] = best_sim
			labeled_rows[i]["probe_mapped"] = int(best_sim >= args.tfidf_threshold)
			if best_sim >= args.tfidf_threshold:
				mapped_any += 1
				mapped_counts[best_bias] += 1

		for b in bias_types:
			tfidf_prev.append({
				"probe_bias_type": b,
				"n_prompts": n,
				"mapped_count": mapped_counts[b],
				"mapped_pct": (100.0 * mapped_counts[b] / n) if n else 0.0,
				"threshold": args.tfidf_threshold,
			})
		tfidf_prev.append({
			"probe_bias_type": "ANY_PROBE_CUE",
			"n_prompts": n,
			"mapped_count": mapped_any,
			"mapped_pct": (100.0 * mapped_any / n) if n else 0.0,
			"threshold": args.tfidf_threshold,
		})
	else:
		for r in labeled_rows:
			r["probe_best_bias"] = ""
			r["probe_best_sim"] = ""
			r["probe_mapped"] = 0

	write_csv(os.path.join(args.out_dir, "prevalence_similarity_tfidf.csv"), tfidf_prev,
			  ["probe_bias_type", "n_prompts", "mapped_count", "mapped_pct", "threshold"])

	# AI labeling
	ai_rows = [{"ai_best_bias": "", "ai_best_score": 0.0, "ai_mapped": 0, "ai_top": []} for _ in cleaned]
	ai_prev = []
	if args.ai_method != "none":
		if args.ai_method == "sbert_sim":
			ai_rows = ai_map_sbert_similarity(cleaned, cue_docs, args.ai_model, args.ai_threshold, topk=args.ai_topk, device=args.device)
		elif args.ai_method == "sbert_clf":
			clf, le, kept_classes, clf_metrics = train_probe_sbert_classifier(
				probe_pairs,
				args.ai_model,
				train_text=args.clf_train_text,
				min_samples_per_class=args.clf_min_samples_per_class,
				class_weight=args.clf_class_weight,
				device=args.device,
				eval_mode=args.clf_eval,
				eval_test_size=args.clf_eval_test_size,
				eval_seed=args.clf_eval_seed,
				eval_threshold=args.ai_threshold,
				return_metrics=True,
			)
			if clf is None:
				print("[WARN] sbert_clf could not be trained (insufficient classes). AI mapping disabled.")
				ai_rows = [{"ai_best_bias": "", "ai_best_score": 0.0, "ai_mapped": 0, "ai_top": []} for _ in cleaned]
			else:
				ai_rows = ai_predict_probe_sbert_classifier(cleaned, clf, le, args.ai_model, args.ai_threshold, topk=args.ai_topk, device=args.device)

				# Save classifier quality stats (precision/recall/F1) on a PROBE holdout split
				if clf_metrics:
					qpath = os.path.join(args.out_dir, "classifier_quality.csv")
					write_csv(qpath, [clf_metrics], list(clf_metrics.keys()))
					if str(clf_metrics.get("note", "")) == "":
						try:
							print(
								"Classifier quality (PROBE holdout) "
								f"precision_macro={float(clf_metrics.get('precision_macro')):.4f} "
								f"recall_macro={float(clf_metrics.get('recall_macro')):.4f} "
								f"f1_macro={float(clf_metrics.get('f1_macro')):.4f}"
							)
						except Exception:
							print(f"Classifier quality saved to: {qpath}")


		elif args.ai_method == "hf_clf":
			hf_out = resolve_hf_out_dir(args, probe_pairs, default_out_dir=os.path.join(args.out_dir, "_hf_bias_clf"))
			model, tokenizer, le, kept_classes, clf_metrics = train_probe_hf_classifier(
				probe_pairs,
				args.hf_model,
				hf_out,
				train_text=args.clf_train_text,
				min_samples_per_class=args.clf_min_samples_per_class,
				eval_mode=args.clf_eval,
				eval_test_size=args.clf_eval_test_size,
				eval_seed=args.clf_eval_seed,
				max_len=args.hf_max_len,
				lr=args.hf_lr,
				epochs=args.hf_epochs,
				train_batch_size=args.hf_train_batch_size,
				eval_batch_size=args.hf_eval_batch_size,
				weight_decay=args.hf_weight_decay,
				warmup_ratio=args.hf_warmup_ratio,
				grad_accum=args.hf_grad_accum,
				fp16=args.hf_fp16,
				num_workers=args.hf_num_workers,
				device=args.device,
				retrain=args.hf_retrain,
				eval_threshold=args.ai_threshold,
				return_metrics=True,
			)
			if model is None:
				print("[WARN] hf_clf could not be trained (insufficient classes). AI mapping disabled.")
				ai_rows = [{"ai_best_bias": "", "ai_best_score": 0.0, "ai_mapped": 0, "ai_top": []} for _ in cleaned]
			else:
				ai_rows = ai_predict_probe_hf_classifier(
					cleaned,
					model,
					tokenizer,
					le,
					args.ai_threshold,
					topk=args.ai_topk,
					device=args.device,
					batch_size=args.hf_eval_batch_size,
					max_len=args.hf_max_len,
					fp16=args.hf_fp16,
				)
				# Save classifier quality stats (PROBE holdout)
				if clf_metrics:
					qpath = os.path.join(args.out_dir, "classifier_quality.csv")
					write_csv(qpath, [clf_metrics], list(clf_metrics.keys()))
					if str(clf_metrics.get("note", "")) == "":
						try:
							print(
								"Classifier quality (PROBE holdout) "
								f"precision_macro={float(clf_metrics.get('precision_macro')):.4f} "
								f"recall_macro={float(clf_metrics.get('recall_macro')):.4f} "
								f"f1_macro={float(clf_metrics.get('f1_macro')):.4f}"
							)
						except Exception:
							print(f"Classifier quality saved to: {qpath}")

	# merge AI rows
	for i in range(len(labeled_rows)):
		labeled_rows[i].update(ai_rows[i])

	# AI prevalence by bias type
	ai_counts = Counter()
	ai_any = 0
	for r in labeled_rows:
		if int(r.get("ai_mapped", 0)) == 1:
			ai_any += 1
			ai_counts[str(r.get("ai_best_bias", ""))] += 1
	for b in bias_types:
		ai_prev.append({
			"ai_bias_type": b,
			"n_prompts": n,
			"mapped_count": ai_counts[b],
			"mapped_pct": (100.0 * ai_counts[b] / n) if n else 0.0,
			"threshold": args.ai_threshold,
			"ai_method": args.ai_method,
			"ai_model": _ai_model_name_for_outputs(args),
		})
	ai_prev.append({
		"ai_bias_type": "ANY_AI_CUE",
		"n_prompts": n,
		"mapped_count": ai_any,
		"mapped_pct": (100.0 * ai_any / n) if n else 0.0,
		"threshold": args.ai_threshold,
		"ai_method": args.ai_method,
		"ai_model": _ai_model_name_for_outputs(args),
	})
	write_csv(os.path.join(args.out_dir, "prevalence_ai.csv"), ai_prev,
			  ["ai_bias_type", "n_prompts", "mapped_count", "mapped_pct", "threshold", "ai_method", "ai_model"])


	# # Example sampling (mapped prompts) for qualitative evidence
	# examples = []
	# per_bias = defaultdict(list)
	# for r in labeled_rows:
	# 	# prefer AI mapping, fall back to TF-IDF mapping
	# 	if int(r.get("ai_mapped", 0)) == 1:
	# 		per_bias[str(r.get("ai_best_bias", "unknown"))].append(r)
	# 	elif int(r.get("probe_mapped", 0)) == 1:
	# 		per_bias[str(r.get("probe_best_bias", "unknown"))].append(r)

	# rng = random.Random(1337)
	# for b, rows_b in per_bias.items():
	# 	rng.shuffle(rows_b)
	# 	for r in rows_b[: args.max_examples_per_bias]:
	# 		examples.append({
	# 			"id": r["id"],
	# 			"ai_best_bias": r.get("ai_best_bias", ""),
	# 			"ai_best_score": r.get("ai_best_score", 0.0),
	# 			"probe_best_bias": r.get("probe_best_bias", ""),
	# 			"probe_best_sim": r.get("probe_best_sim", 0.0),
	# 			"prompt_excerpt": (r["prompt_clean"][:280] + ("…" if len(r["prompt_clean"]) > 280 else "")),
	# 			"source_file": r["source_file"],
	# 		})
	# write_jsonl(os.path.join(args.out_dir, "examples.jsonl"), examples)

	# Write labeled prompts CSV (include new columns)
	if labeled_rows:
		base_fields = ["id", "source_file", "source_index", "decision_like", "prompt_clean", "prompt_raw"]
		mapping_fields = [
			"probe_mapped", "probe_best_bias", "probe_best_sim",
			"ai_mapped", "ai_best_bias", "ai_best_score",
		]
		feat_fields = []
		for feat in FEATURES:
			feat_fields += [f"{feat}_strong", f"{feat}_weak", f"{feat}_any"]
		# also store ai_top as json string
		for r in labeled_rows:
			r["ai_top_json"] = json.dumps(r.get("ai_top", []), ensure_ascii=False)
		extra_fields = ["ai_top_json"]

		ordered = []
		for f in base_fields + mapping_fields + feat_fields + extra_fields:
			if f in labeled_rows[0] and f not in ordered:
				ordered.append(f)
		for f in labeled_rows[0].keys():
			if f not in ordered:
				ordered.append(f)

		write_csv(os.path.join(args.out_dir, "devgpt_labeled.csv"), labeled_rows, ordered)

	# Manual validation actions
	if args.manual_mode == "create":
		eval_rows = make_manual_eval_set(labeled_rows)
		if not eval_rows:
			print("[WARN] manual_mode=create: no rows passed the AI threshold (ai_mapped==1); nothing to annotate.")
		else:
			blind_path = write_manual_validation_files(args.out_dir, eval_rows, include_predictions=False, fname="manual_evaluation_blind.csv")
			pred_path = write_manual_validation_files(args.out_dir, eval_rows, include_predictions=True, fname="manual_evaluation_with_predictions.csv")
			print(f"Manual evaluation set created ({len(eval_rows)} rows):\n  {blind_path}\n  {pred_path}")
			print("Annotate the BLIND CSV (fill ann1_* and optionally ann2_*), then run --manual_mode evaluate.")

	elif args.manual_mode == "evaluate":
		if not args.manual_annotated_csv:
			print("[ERROR] manual_mode=evaluate requires --manual_annotated_csv path.")
		else:
			labeled_csv = os.path.join(args.out_dir, "devgpt_labeled.csv")
			note = f"AI: {args.ai_method} | model={args.ai_model} | ai_threshold={args.ai_threshold} | tfidf_threshold={args.tfidf_threshold}"
			metrics_csv, report_txt = evaluate_manual_annotations(args.out_dir, args.manual_annotated_csv, labeled_csv, threshold_note=note)
			print(f"Manual evaluation evaluated:\n  {metrics_csv}\n  {report_txt}")

	# Console summary
	print(f"Saved outputs to: {args.out_dir}")
	print(f"DevGPT prompts analyzed: {n} (subset={args.subset})")
	print(f"PROBE pairs found: {len(probe_pairs)}; cue docs: {len(cue_docs)}")
	print(f"AI method: {args.ai_method} (threshold={args.ai_threshold})")
	if args.ai_method != "none":
		dev = pick_device(args.device)
		print(f"SBERT device: {dev} (requested={args.device})")
	print(f"TF-IDF threshold: {args.tfidf_threshold}")

if __name__ == "__main__":
	main()
