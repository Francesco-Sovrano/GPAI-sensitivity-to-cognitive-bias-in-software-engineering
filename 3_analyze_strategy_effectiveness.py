import os
import json
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns  # compact for heatmaps
from scipy.stats import wilcoxon, mannwhitneyu, ranksums

# -----------------------------
# Matplotlib for figures
# -----------------------------
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

DEFAULT_FONTSIZE = 9
DEFAULT_ALTERNATIVE = "two-sided"
# DEFAULT_ALTERNATIVE = "less"

# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Analyze bias sensitivity across prompting strategies, aggregated by strategy (not model).")
parser.add_argument("--self_assessment", action="store_true")
parser.add_argument("--seed_corpus_only", action="store_true", help="Restrict evaluation to only the human-seeded dilemmas (filtering out AI-generated ones)")
parser.add_argument("--show_figures", action="store_true", help="Show figures after saving (default: False)")
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=0)
parser.add_argument(
	"--sort_biases", choices=["original","alpha"], default="alpha", help=(
		"Ordering of bias facets: original (file order), alpha (alphabetical), "
		"sensitivity (descending mean sensitivity across models)"
	)
)
parser.add_argument("--results_dir", type=str, default="./generated_output_data/", help="Directory containing JSON results files.")
parser.add_argument("--outdir", type=str, default="./strategy_effectiveness_analyses/", help="Directory to write analysis CSVs/figures to.")
parser.add_argument("--strategies", nargs="*", default=None, help=("Optional subset of strategy short labels to analyze. Defaults to all defined."))
parser.add_argument("--bootstrap", type=int, default=2000, help="Number of bootstrap resamples for confidence intervals (default: 2000).")
parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping.")
parser.add_argument("--skip_model_breakouts", action="store_true", default=True, help="Skip model-level pairwise stats (default: True)")

args = parser.parse_args()

# -----------------------------
# Model Name Mapping & Ordering (used only to locate files)
# -----------------------------
model_mapping = {
	"gpt-4.1-nano": "gpt-4.1-nano",
	"gpt-4.1-mini": "gpt-4.1-mini",
	"gpt-4o-mini": "gpt-4o-mini",
	"llama-3.1-8b-instant": "llama-3.1",
	"llama-3.3-70b-versatile": "llama-3.3",
	"deepseek-r1-distill-llama-70b": "deepseek-r1",
}

# -----------------------------
# Prompting Strategy Mapping (no CLI flags!)
# -----------------------------
# short label → list of filename flag fragments appended as ".{flag}=True"
NO_STRATEGY_LABEL = "Ø"
PROBE_AXIOMS_INJECTION_LABEL = 'ProbeAX'
strategy_mapping = {
	## No strategy
	NO_STRATEGY_LABEL: [],  # no strategy
	PROBE_AXIOMS_INJECTION_LABEL: ["inject_axioms"], # cueing with axioms (not a debiasing strategy)
	## Baseline prompting strategies
	"CoT": ["chain_of_thought"],
	"IMP": ["implication_prompting"],
	"IsD": ["impersonified_self_debiasing"],
	"BW": ["bias_warning"],
	"BW+IsD": ["bias_warning", "impersonified_self_debiasing"],
	## Our prompting strategies
	"2sAX": ["bistep_axioms_elicitation"],
	"2sAX+BW": ["bias_warning", "bistep_axioms_elicitation"],
	"sAX": ["self_axioms_elicitation"],
	"sAX+BW": ["bias_warning", "self_axioms_elicitation"],
	"sAX+BW+IsD": ["bias_warning", "impersonified_self_debiasing", "self_axioms_elicitation"],
	# "sAX+BW+2sAX": ["bias_warning", "bistep_axioms_elicitation", "self_axioms_elicitation"],
	# "AllGood": ["bias_warning", "impersonified_self_debiasing", "bistep_axioms_elicitation", "self_axioms_elicitation"],
}

# -----------------------------
# Data Loading
# -----------------------------

def load_bias_json(json_file_path):
	with open(json_file_path, 'r') as f:
		data = json.load(f)
	df = pd.DataFrame.from_dict(data, orient='index').reset_index()
	df = df.rename(columns={'index': 'bias'})
	return df


def _pattern_for(model_key, strategy_flags: list) -> str:
	base = f"1_bias_sensitivity_model={model_key}"
	if not args.self_assessment:
		base += ".data_model_list=[[]*[]]"
	if args.seed_corpus_only:
		base += ".seed_corpus_only=True"
	for flag in strategy_flags:
		base += f".{flag}=True"
	if args.temperature:
		base += f".temperature={args.temperature}.top_p={args.top_p}.json"
	else:
		base += ".json"
	return base


def load_all_results(results_dir):
	base = Path(results_dir)
	df_list = []
	strategies_to_use = args.strategies or list(strategy_mapping.keys())

	for model_key, model_disp in model_mapping.items():
		for strat_label in strategies_to_use:
			if strat_label not in strategy_mapping:
				print(f"[WARN] Unknown strategy label '{strat_label}' – skipping.")
				continue
			pattern = _pattern_for(model_key, strategy_mapping[strat_label])
			path_list = list(map(lambda p: p.name, base.glob(pattern)))
			if not path_list:
				print(f"[MISS] No files for model={model_key}, strategy={strat_label} (pattern: {pattern})")
				continue
			json_file = os.path.join(results_dir, sorted(path_list, key=len, reverse=True)[0])
			try:
				df = load_bias_json(json_file)
			except Exception as e:
				print(f"[ERROR] Failed to load {json_file}: {e}")
				continue
			df["model_key"] = model_key
			df["model"] = model_disp
			df["strategy"] = strat_label
			# ensure expected columns exist
			for col in ["sensitivity", "harmfulness"]:
				if col not in df.columns:
					df[col] = np.nan
			df_list.append(df)

	if df_list:
		return pd.concat(df_list, ignore_index=True)
	else:
		raise FileNotFoundError("No JSON files found matching any model × strategy combination.")

# -----------------------------
# Complexity expansion & aggregation
# -----------------------------

def expand_complexity_tiers(df):
	records = []
	for _, row in df.iterrows():
		comp = row.get('complexity_analysis', {}) or {}
		if not isinstance(comp, dict):
			continue
		for tier, metrics in comp.items():
			if not isinstance(metrics, dict):
				continue
			records.append({
				'bias': row['bias'],
				'model': row['model'],
				'strategy': row['strategy'],
				'tier': tier,
				'tier_total_cases': metrics.get('total_cases', np.nan),
				'tier_sensitivity': metrics.get('sensitivity', np.nan),
				'tier_harmfulness': metrics.get('harmfulness', np.nan),
				'tier_prolog_uncertainty': metrics.get('prolog_uncertainty', np.nan),
			})
	if not records:
		return pd.DataFrame(columns=['bias','model','strategy','tier','tier_total_cases','tier_sensitivity','tier_harmfulness','tier_prolog_uncertainty'])
	long_df = pd.DataFrame.from_records(records)
	tier_order = ['low', 'mid-low', 'mid-high', 'high']
	long_df['tier'] = pd.Categorical(long_df['tier'], categories=tier_order, ordered=True)
	return long_df

# -----------------------------
# Plotting helpers (matplotlib only, no seaborn)
# -----------------------------

def _ensure_outdir(d):
	d.mkdir(parents=True, exist_ok=True)

def _ordered_strategy_columns(cols):
	"""Return strategy columns ordered like strategy_mapping (respecting --strategies if given)."""
	base = list(strategy_mapping.keys())
	if args.strategies:
		# keep dict order but filter to the selected subset
		base = [s for s in base if s in args.strategies]
	# keep only those that are actually present in cols
	return [s for s in base if s in cols]

# -----------------------------
# Aggregations focused on STRATEGY (not model)
# -----------------------------

def aggregate_strategy_overall(df):
	# Average across models within each bias × strategy, then average across biases for strategy-level mean
	per_bias_strat = (df.dropna(subset=['sensitivity'])
						.groupby(['bias','strategy'])['sensitivity']
						.mean()
						.reset_index())
	overall = (per_bias_strat.groupby('strategy')['sensitivity']
			   .agg(['mean','count'])
			   .reset_index()
			   .rename(columns={'mean':'mean_sensitivity','count':'n_biases'}))
	return overall.sort_values('mean_sensitivity', ascending=False)


def aggregate_strategy_by_tier(long_df):
	# Average across models within each bias × strategy × tier, then average across biases
	per_bias = (long_df.dropna(subset=['tier_sensitivity'])
						.groupby(['bias','strategy','tier'])['tier_sensitivity']
						.mean()
						.reset_index())
	out = (per_bias.groupby(['strategy','tier'])['tier_sensitivity']
		   .agg(['mean','count'])
		   .reset_index()
		   .rename(columns={'mean':'mean_sensitivity','count':'n_biases'}))
	return out.sort_values(['tier','mean_sensitivity'], ascending=[True, False])

def compute_rbs(_x, _y, stat):
	return 1 - (2 * stat / (len(_x) * len(_y)))

def benjamini_hochberg(pvals):
	"""
	Benjamini–Hochberg FDR correction.

	Parameters
	----------
	pvals : array-like
		Sequence of p-values.

	Returns
	-------
	np.ndarray
		FDR-adjusted p-values (q-values), same shape/order as input.
	"""
	pvals = np.asarray(pvals, dtype=float)
	m = pvals.size
	if m == 0:
		return pvals

	order = np.argsort(pvals)
	ranked = pvals[order]

	adjusted = np.empty(m, dtype=float)
	prev = 1.0
	for i in range(m - 1, -1, -1):
		p = ranked[i]
		q = p * m / (i + 1.0)
		if q > prev:
			q = prev
		prev = q
		adjusted[i] = q

	adjusted = np.minimum(adjusted, 1.0)
	out = np.empty(m, dtype=float)
	out[order] = adjusted
	return out

def plot_box_by_strategy(
	aggregated_samples_df, samples_df,
	label_col, unit_col, value_col,
	title, outpath,
	fontsize=DEFAULT_FONTSIZE,
	outpath_stats_csv=None,
):
	# --- Collect basic per-label stats into dicts ---
	unique_values = aggregated_samples_df[label_col].dropna().unique().tolist()

	data, means, medians = {}, {}, {}
	for s in unique_values:
		vals = (
			aggregated_samples_df.loc[aggregated_samples_df[label_col] == s, value_col]
			.astype(float)
			.to_numpy()
		)
		vals = vals[~np.isnan(vals)]
		if vals.size == 0:
			vals = np.array([np.nan])  # placeholder
		data[s] = vals
		means[s] = np.nanmean(vals)
		medians[s] = np.nanmedian(vals)

	# --- Mann-Whitney U + rank biserial effect size (strategy vs Ø) ---
	rbs_results = {}
	n_results = {}

	# IMPORTANT FIX: collapse raw samples to 1 value per (unit_col, strategy) before testing
	_collapsed = (
		samples_df[[unit_col, label_col, value_col]]
		.dropna()
		.groupby([unit_col, label_col], as_index=False)[value_col]
		.mean()
	)

	for s in unique_values:
		if s == NO_STRATEGY_LABEL:
			rbs_results[s] = {"p": np.nan, "rbs": 0.0, "baseline": ""}
			n_results[s] = {"n_units": np.nan}
			continue

		df_s = _collapsed[_collapsed[label_col] == s][[unit_col, value_col]].rename(columns={value_col: "val_s"})

		# Global baseline: always compare every method to Ø
		baseline = NO_STRATEGY_LABEL

		df_b = _collapsed[_collapsed[label_col] == baseline][[unit_col, value_col]].rename(columns={value_col: "val_b"})
		merged = pd.merge(df_s, df_b, on=unit_col, how="inner").sort_values(by=unit_col)
		_x = merged["val_s"].to_numpy()
		_y = merged["val_b"].to_numpy()

		n_results[s] = {"n_units": int(len(merged))}

		if len(_x) == 0 or len(_y) == 0:
			rbs_results[s] = {"p": np.nan, "rbs": np.nan, "baseline": baseline}
			continue

		try:
			stat, p = mannwhitneyu(_x, _y, alternative=DEFAULT_ALTERNATIVE)
			rbs = compute_rbs(_x, _y, stat)
			rbs_results[s] = {"p": float(p), "rbs": float(rbs), "u": float(stat), "baseline": baseline}
		except ValueError:
			rbs_results[s] = {"p": np.nan, "rbs": np.nan, "baseline": baseline}

	# --- Benjamini–Hochberg FDR correction (all methods vs Ø) ---
	raw_ps = []
	labels_for_bh = []
	for s, res in rbs_results.items():
		if s == NO_STRATEGY_LABEL:
			continue
		p = res.get("p")
		if isinstance(p, float) and np.isfinite(p):
			labels_for_bh.append(s)
			raw_ps.append(p)

	if raw_ps:
		qvals = benjamini_hochberg(raw_ps)
		for s, q in zip(labels_for_bh, qvals):
			rbs_results[s]["p_adj"] = float(q)
# --- OPTIONAL: write stats CSV ---
	if outpath_stats_csv:
		rows = []
		for s in sorted(unique_values, key=lambda k: (k != NO_STRATEGY_LABEL, k)):
			res = rbs_results.get(s, {})
			rows.append({
				"strategy": s,
				"baseline": res.get("baseline", NO_STRATEGY_LABEL),
				"alternative": DEFAULT_ALTERNATIVE,
				"n_units_aligned": n_results.get(s, {}).get("n_units", np.nan),
				"u_stat": res.get("u", np.nan),
				"p_raw": res.get("p", np.nan),
				"p_fdr": res.get("p_adj", np.nan),
				"r_rb": res.get("rbs", np.nan),
				"mean_display_vals": means.get(s, np.nan),      # from aggregated_samples_df
				"median_display_vals": medians.get(s, np.nan),  # from aggregated_samples_df
			})
		pd.DataFrame(rows).to_csv(outpath_stats_csv, index=False)

	# --- Sort labels by ascending effect size (RBS) ---
	def rbs_key(s):
		if s == NO_STRATEGY_LABEL:
			return 0
		v = rbs_results.get(s, {}).get("rbs")
		return v if isinstance(v, (float, int)) and np.isfinite(v) else np.inf

	labels = sorted(unique_values, key=rbs_key)

	# --- Plot ---
	plt.figure(figsize=(len(labels)*1.1, 4), constrained_layout=True)
	x = np.arange(len(labels))

	# prepare box colors: highlight NO_STRATEGY_LABEL
	box_colors = ["lightgray" if s == NO_STRATEGY_LABEL else "white" for s in labels]

	bp = plt.boxplot(
		[data[s] for s in labels],
		positions=x,
		widths=0.6,
		patch_artist=True,   # so we can color boxes
		manage_ticks=False,
		showfliers=True,
	)
	for patch, color in zip(bp["boxes"], box_colors):
		patch.set_facecolor(color)
		patch.set_alpha(0.6 if color != "white" else 1.0)
		patch.set_edgecolor("black")

	# Axis cosmetics
	plt.xticks(x, labels, rotation=0)
	plt.ylabel("Sensitivity")
	plt.title(title)

	# Accessibility: use typography (bold) rather than color for special columns
	ax = plt.gca()
	for c in labels:
		if c in [PROBE_AXIOMS_INJECTION_LABEL, NO_STRATEGY_LABEL]:
			idx = labels.index(c)
			ax.get_xticklabels()[idx].set_fontweight("bold")
		elif 'AX' in c:
			idx = labels.index(c)
			ax.get_xticklabels()[idx].set_fontweight("bold")

	# --- Compute axis limits and spacing (robust to empty/NaN-only groups) ---
	finite_vals = []
	for v in data.values():
		if v is None or np.size(v) == 0:
			continue
		v = np.asarray(v, dtype=float)
		v = v[np.isfinite(v)]
		if v.size:
			finite_vals.append(v)
	if finite_vals:
		allv = np.concatenate(finite_vals)
		data_min = float(np.min(allv))
		data_max = float(np.max(allv))
	else:
		data_min, data_max = 0.0, 1.0
	y_range = (data_max - data_min) if np.isfinite(data_max) and np.isfinite(data_min) and data_max != data_min else 1.0
	y_offset_mean = 0.01 * y_range
	y_offset_median = 0.01 * y_range
	y_offset_stats = 0.06 * y_range
	plt.ylim(top=(data_max + 0.10 * y_range), bottom=(data_min - 2 * y_offset_stats))
	ax = plt.gca()
	ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

	# --- Mean and median annotations ---
	for i, s in enumerate(labels):
		m = means[s]
		if np.isfinite(m):
			plt.plot(x[i], m, marker="D", color="black")
			plt.text(x[i], m + y_offset_mean, f"{m:.2f}%", ha="center", va="bottom", fontsize=fontsize,
				bbox=dict(
						boxstyle="round,pad=0.2",
						facecolor="white",
						edgecolor="none",
						alpha=0.9
					))

		vals = data[s][~np.isnan(data[s])]
		if vals.size:
			med = medians[s]
			plt.plot(x[i], med, marker="o", color="red")
			plt.text(x[i], med + y_offset_median, f"{med:.2f}%", ha="center", va="bottom", fontsize=fontsize,
				bbox=dict(
						boxstyle="round,pad=0.2",
						facecolor="white",
						edgecolor="none",
						alpha=0.9
					))

	# --- Annotate p-values (FDR-corrected) and RBS ---
	alpha = 0.05
	for i, s in enumerate(labels):
		if s == NO_STRATEGY_LABEL:
			continue
		stats_result = rbs_results.get(s, {})
		p = stats_result.get("p", "")
		p_adj = stats_result.get("p_adj", None)
		rbs = stats_result.get("rbs", "")

		if isinstance(p, float):
			# use FDR-adjusted p when available
			p_for_sig = p_adj if isinstance(p_adj, float) else p
			label = "p(FDR)" if isinstance(p_adj, float) else "p"

			if p_for_sig < 0.001:
				p_text = f"{label}<0.001"
			else:
				p_text = f"{label}={p_for_sig:.3f}"

			if isinstance(rbs, (float, int)) and np.isfinite(rbs):
				rbs_text = fr"$\bf{{r_{{rb}}\!=\!{rbs:.2f}}}$" if p_for_sig < alpha else fr"$r_{{rb}}={rbs:.2f}$"
			else:
				rbs_text = r"$r_{rb}=\text{n/a}$"

			fontweight = "bold" if p_for_sig < alpha else "normal"
		elif p in ["-", "n/a", "err"]:
			p_text = f"p={p}"
			rbs_text = fr"$r_{{rb}}={rbs}$"
			fontweight = "normal"
		else:
			p_text = "p=?"
			rbs_text = r"$r_{{rb}}=?$"
			fontweight = "normal"

		plt.text(
			x[i],
			data_min - y_offset_stats / 3,
			f"{p_text}\n{rbs_text}",
			ha="center",
			va="top",
			fontsize=fontsize,
			fontweight=fontweight,
			bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
		)

	plt.tight_layout()
	plt.savefig(outpath, dpi=200, bbox_inches="tight", pad_inches=0)
	if args.show_figures:
		plt.show()
	plt.close()


def plot_heatmap(
	df,
	outpath_pdf,
	*,
	row_col,
	col_col="strategy",
	value_col="sensitivity",
	samples_df=None,
	baseline_col_value=NO_STRATEGY_LABEL,
	exclude_from_best=None,
	pre_agg_cols=None,
	pre_agg_func="mean",
	outpath_csv=None,
	outpath_stats_csv=None,
	outpath_pvals_csv=None,
	outpath_qvals_csv=None,
	split_strategy_groups=False,
	# Optional titles for the two strategy slices (RQ1 vs RQ2)
	split_titles=("RQ1: Baseline strategies", "RQ2: Proposed AX strategies"),
	cmap="viridis_r",
	center=None,
	fontsize=12,
	gap_width=0.06,
	show=False
):
	"""
	Universal heatmap with per-row BH/FDR:
	  - Per-row tests: strategy vs baseline_col_value.
	  - BH FDR correction applied within each row's family of tests.
	  - Stars in cells reflect FDR-adjusted p.
	"""
	# ---------- prep & optional pre-aggregation ----------
	work = df.copy()
	if pre_agg_cols:
		work = (work.dropna(subset=[value_col])
					.groupby(pre_agg_cols, as_index=False)[value_col]
					.agg(pre_agg_func))

	# ---------- pivot ----------
	pivot_df = (
		work.dropna(subset=[value_col])
			.groupby([row_col, col_col], as_index=False)[value_col]
			.median()
			.pivot(index=row_col, columns=col_col, values=value_col)
	)
	if pivot_df.size == 0:
		raise ValueError("No data to plot after grouping/pivot. Check your row_col/col_col/value_col.")

	# ---------- column ordering & special placement ----------
	pivot_df = pivot_df.reindex(columns=_ordered_strategy_columns(pivot_df.columns))

	# --- Ensure NO_STRATEGY_LABEL is the last column ---
	if baseline_col_value in pivot_df.columns:
		pivot_df = pivot_df[[c for c in pivot_df.columns if c != baseline_col_value] + [baseline_col_value]]
	if PROBE_AXIOMS_INJECTION_LABEL in pivot_df.columns:
		pivot_df = pivot_df[[PROBE_AXIOMS_INJECTION_LABEL] + [c for c in pivot_df.columns if c != PROBE_AXIOMS_INJECTION_LABEL]]

	# Optional CSV
	if outpath_csv:
		pivot_df.to_csv(outpath_csv)

	
	# Strategy grouping (used for slicing and for selecting the appropriate baseline in tests)
	_RQ2_PROPOSED = ["2sAX", "2sAX+BW", "sAX", "sAX+BW", "sAX+BW+IsD"]

	def _baseline_for_col(col_name):
		"""Return the baseline strategy to compare against for this column, or None if not applicable.

		User-specified baseline logic:
		- Ø (empty set) is the single global baseline and is never tested (it would be compared to itself).
		- Every other method (including ProbeAX and all AX variants) is compared to Ø.
		"""
		if col_name == baseline_col_value:
			return None
		return baseline_col_value

		if col_col == "strategy" and col_name in _RQ2_PROPOSED and PROBE_AXIOMS_INJECTION_LABEL in pivot_df.columns:
			return PROBE_AXIOMS_INJECTION_LABEL
		return baseline_col_value

	# ---------- per-row p-values & rank-biserial effect sizes ----------
	pval_dict = {}
	if samples_df is not None:
		for r in pivot_df.index:
			for c in pivot_df.columns:
				baseline_name = _baseline_for_col(c)
				if baseline_name is None:
					pval_dict[(r, c)] = (None, 0)
					continue
				vals_s = samples_df[(samples_df[row_col] == r) & (samples_df[col_col] == c)][value_col].dropna()
				vals_b = samples_df[(samples_df[row_col] == r) & (samples_df[col_col] == baseline_name)][value_col].dropna()
				if len(vals_s) > 0 and len(vals_b) > 0:
					try:
						stat, p = mannwhitneyu(vals_s, vals_b, alternative=DEFAULT_ALTERNATIVE)
						rbs = compute_rbs(vals_s, vals_b, stat)
						pval_dict[(r, c)] = (p, rbs)
					except Exception:
						pval_dict[(r, c)] = (None, 0)
				else:
					pval_dict[(r, c)] = (None, 0)

	# ---------- Benjamini–Hochberg FDR correction per row ----------
	qval_dict = {}
	if samples_df is not None:
		for r in pivot_df.index:
			row_ps = []
			cols_for_row = []
			for c in pivot_df.columns:
				# skip columns that are not tested (their baseline is None)
				if _baseline_for_col(c) is None:
					qval_dict[(r, c)] = None
					continue
				p = pval_dict.get((r, c), (None, 0))[0]
				if isinstance(p, float) and np.isfinite(p):
					row_ps.append(p)
					cols_for_row.append(c)
			if row_ps:
				qvals = benjamini_hochberg(row_ps)
				for c, q in zip(cols_for_row, qvals):
					qval_dict[(r, c)] = q

	# ---------- OPTIONAL: export p-values/effect sizes ----------
	if samples_df is not None and (outpath_stats_csv or outpath_pvals_csv or outpath_qvals_csv):
		stats_rows = []
		for r in pivot_df.index:
			for c in pivot_df.columns:
				baseline_name = _baseline_for_col(c)
				n_s = int(samples_df[(samples_df[row_col] == r) & (samples_df[col_col] == c)][value_col].dropna().shape[0])

				if baseline_name is None:
					n_b = 0
					base_median = np.nan
				else:
					n_b = int(samples_df[(samples_df[row_col] == r) & (samples_df[col_col] == baseline_name)][value_col].dropna().shape[0])
					base_median = pivot_df.loc[r, baseline_name] if baseline_name in pivot_df.columns else np.nan

				p_raw, rbs = pval_dict.get((r, c), (None, np.nan))
				q = qval_dict.get((r, c), None)
				cell_median = pivot_df.loc[r, c]

				stats_rows.append({
					row_col: r,
					col_col: c,
					"baseline": baseline_name if baseline_name is not None else "",
					"alternative": DEFAULT_ALTERNATIVE,
					"n_strategy": n_s,
					"n_baseline": n_b,
					"median_strategy": float(cell_median) if pd.notna(cell_median) else np.nan,
					"median_baseline": float(base_median) if pd.notna(base_median) else np.nan,
					"delta_median": (float(cell_median) - float(base_median)) if (pd.notna(cell_median) and pd.notna(base_median)) else np.nan,
					"p_raw": float(p_raw) if isinstance(p_raw, float) and np.isfinite(p_raw) else np.nan,
					"p_fdr_row": float(q) if isinstance(q, float) and np.isfinite(q) else np.nan,
					"r_rb": float(rbs) if isinstance(rbs, (float, int)) and np.isfinite(rbs) else np.nan,
				})

		stats_df = pd.DataFrame(stats_rows)

		if outpath_stats_csv:
			stats_df.to_csv(outpath_stats_csv, index=False)

		if outpath_pvals_csv:
			p_wide = stats_df.pivot(index=row_col, columns=col_col, values="p_raw")
			p_wide.to_csv(outpath_pvals_csv)

		if outpath_qvals_csv:
			q_wide = stats_df.pivot(index=row_col, columns=col_col, values="p_fdr_row")
			q_wide.to_csv(outpath_qvals_csv)

	# ---------- best-cell mask (min metric; tie-break max |r_rb|) ----------
	if exclude_from_best is None:
		exclude_from_best = [baseline_col_value]
	best_mask = pivot_df.notna() & False
	effect_df = pivot_df.copy().astype(float)
	effect_df.loc[:, :] = float("nan")

	if samples_df is not None:
		for r in pivot_df.index:
			for c in pivot_df.columns:
				if c in exclude_from_best:
					continue
				p, rbs = pval_dict.get((r, c), (None, float("nan")))
				effect_df.loc[r, c] = (rbs if p is not None else float("nan"))

	for r in pivot_df.index:
		candidates = [c for c in pivot_df.columns if c not in exclude_from_best]
		row_vals = pivot_df.loc[r, candidates].astype(float)
		if not row_vals.notna().any():
			continue
		min_v = row_vals.min(skipna=True)
		tied = row_vals.index[row_vals == min_v]
		if len(tied) == 1 or samples_df is None:
			chosen = tied[0]
		else:
			eff = effect_df.loc[r, tied].astype(float).abs()
			chosen = eff.idxmax() if eff.notna().any() else tied[0]
		best_mask.loc[r, :] = False
		best_mask.loc[r, chosen] = True

	# ---------- annotations (value% + stars based on FDR) ----------
	ann_df = pivot_df.copy().astype(float)
	for i, r in enumerate(pivot_df.index):
		for j, c in enumerate(pivot_df.columns):
			v = ann_df.loc[r, c]
			if samples_df is not None:
				p_raw, _ = pval_dict.get((r, c), (None, 0))
				q = qval_dict.get((r, c))
				p_use = q if isinstance(q, float) else p_raw
				if isinstance(p_use, float) and np.isfinite(p_use):
					stars = (r"$^{***}$" if p_use < 0.001 else
							 (r"$^{**}$" if p_use < 0.01 else
							  (r"$^{*}$" if p_use < 0.05 else "")))
				else:
					stars = ""
			else:
				stars = ""
			ann_df.loc[r, c] = f"{v:.1f}%{stars}"

	# ---------- color by row-normalized z, label with actual % ----------
	z = (pivot_df - pivot_df.mean(axis=1).values[:, None]) / pivot_df.std(axis=1).replace(0, np.nan).values[:, None]
	vmin = np.nanpercentile(z.values, 20)
	vmax = np.nanpercentile(z.values, 80)

	# Strategy slicing (to reduce density): RQ1 (baseline) vs RQ2 (proposed AX strategies)
	def _strategy_slices(cols):
		# Density reduction layout:
		#   [Ø baseline] | [RQ1 baseline prompting] | [RQ2 AX prompting]
		# Ø is shown once (left) because it is the common baseline for *all* methods.
		baseline_methods = ["CoT", "IMP", "IsD", "BW", "BW+IsD"]
		proposed_methods = [PROBE_AXIOMS_INJECTION_LABEL, "2sAX", "2sAX+BW", "sAX", "sAX+BW", "sAX+BW+IsD"]

		out = []

		# Panel 0 (common baseline)
		if baseline_col_value in cols:
			# out.append(("Baseline (Ø)", [baseline_col_value]))
			out.append(("", [baseline_col_value]))

		# Panel 1 (RQ1)
		s1 = [c for c in baseline_methods if c in cols]
		if s1:
			out.append((split_titles[0] if split_titles else "", s1))

		# Panel 2 (RQ2)
		s2 = [c for c in proposed_methods if c in cols and c != baseline_col_value]
		if s2:
			# If the caller only provided two titles, use the second for RQ2.
			title = split_titles[1] if (split_titles and len(split_titles) > 1) else ""
			out.append((title, s2))

		# Fallback: if slicing isn't meaningful, plot a single heatmap.
		if len(out) < 2:
			out = [("", [c for c in _ordered_strategy_columns(cols)])]
		return out

	def _format_ytick(text):
		if row_col == "tier":
			m = {
				"low": "Low complexity",
				"mid-low": "Mid-low complexity",
				"mid-high": "Mid-high complexity",
				"high": "High complexity",
			}
			text = m.get(text, text)
		return text.replace(" ", "\n")

	def _add_gaps(ax, cols):
		# subtle gaps around Ø and after ProbeAX (if present in this slice)
		if baseline_col_value in cols and len(cols) > 1:
			idx = cols.index(baseline_col_value)
			ax.axvspan(idx - gap_width/2, idx + gap_width/2, color=ax.figure.get_facecolor(), zorder=6, lw=1)
		if PROBE_AXIOMS_INJECTION_LABEL in cols:
			idx = cols.index(PROBE_AXIOMS_INJECTION_LABEL) + 1
			ax.axvspan(idx - gap_width/2, idx + gap_width/2, color=ax.figure.get_facecolor(), zorder=6, lw=1)

	def _bold_special_xticks(ax, cols):
		# Accessibility: use typography (bold) rather than color for special columns.
		for special in [baseline_col_value]:
			if special in cols:
				idx = cols.index(special)
				ax.get_xticklabels()[idx].set_fontweight("bold")

	def _decorate_cells(ax, cols, n_rows):
		# seaborn puts annotation texts first, row-major
		n_cols = len(cols)
		ann_texts = ax.texts[: n_rows * n_cols]
		ann = {(i, j): ann_texts[i * n_cols + j] for i in range(n_rows) for j in range(n_cols)}

		for i, r in enumerate(pivot_df.index):
			for j, c in enumerate(cols):
				p_raw, effect_size = pval_dict.get((r, c), (None, 0))
				q = qval_dict.get((r, c))
				p_use = q if isinstance(q, float) else p_raw

				if c in pivot_df.columns and best_mask.loc[r, c]:
					ann[(i, j)].set_fontweight("bold")

				if samples_df is not None and p_use is not None and isinstance(effect_size, (float, int)):
					if abs(effect_size) >= 0.01:
						p_text = fr"$r_{{rb}}\!=\!{effect_size:.2f}$"
					else:
						p_text = fr"$|r_{{rb}}|\!<\!0.01$"
					ax.text(
						j + 0.5, i + 0.67, p_text,
						ha="center", va="top",
						fontstyle="italic",
						fontweight=ann[(i, j)].get_fontweight(),
						fontsize=fontsize-2,
						color=ann[(i, j)].get_color(),
						path_effects=ann[(i, j)].get_path_effects(),
					)

	# ---------- plot (single or sliced) ----------
	if split_strategy_groups and col_col == "strategy":
		slices = _strategy_slices(list(pivot_df.columns))
		height = max(2.8, len(pivot_df) * 0.55)
		width = sum(len(cols) for _, cols in slices) * 0.75 + 0.8 * (len(slices) - 1)
		fig, axes = plt.subplots(
			1, len(slices),
			figsize=(max(6.5, width), height),
			sharey=False,
			gridspec_kw={"width_ratios": [len(cols) for _, cols in slices], "wspace": 0.08}
		)
		if not isinstance(axes, (list, np.ndarray)):
			axes = [axes]

		for si, (title, cols) in enumerate(slices):
			ax = axes[si]
			z_sub = z[cols]
			ann_sub = ann_df[cols]
			sns.heatmap(
				z_sub,
				annot=ann_sub,
				fmt="",
				vmin=vmin,
				vmax=vmax,
				cmap=cmap,
				center=center,
				cbar=(si == len(slices) - 1),
				cbar_kws={"label": "Row-norm z", "pad": 0.01, "aspect": 30} if (si == len(slices) - 1) else None,
				linewidths=0.05,
				linecolor="gray",
				ax=ax,
				yticklabels=[_format_ytick(str(v)) for v in z_sub.index] if si == 0 else False
			)
			ax.set_ylabel("")
			ax.set_xlabel("")
			ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
			_bold_special_xticks(ax, cols)
			_add_gaps(ax, cols)

			# y tick labels only on the first slice (explicitly set to avoid them disappearing)
			if si == 0:
				ax.tick_params(axis='y', labelleft=True, left=True)
				ax.set_yticks(np.arange(len(z_sub.index)) + 0.5)
				ax.set_yticklabels([_format_ytick(str(v)) for v in z_sub.index], rotation=0)
				ax.tick_params(axis="y", labelsize=fontsize)
			else:
				ax.set_yticklabels([])
				ax.tick_params(axis='y', labelleft=False, left=False)

			if title:
				ax.set_title(title, fontsize=fontsize+1)

			_decorate_cells(ax, cols, n_rows=len(pivot_df.index))

		plt.subplots_adjust(wspace=0.0, left=0.0)
	else:
		plt.figure(figsize=(len(pivot_df.columns) * .9, max(2.8, len(pivot_df) * 0.55)))
		ax = sns.heatmap(
			z,
			annot=ann_df,
			fmt="",
			vmin=vmin,
			vmax=vmax,
			cmap=cmap,
			center=center,
			cbar_kws={"label": "Row-norm z", "pad": 0.01, "aspect": 30},
			linewidths=0.5,
			linecolor="gray"
		)
		ax.set_ylabel("")
		ax.set_xlabel("")
		plt.xticks(rotation=25, ha="right")
		ax.set_yticklabels([_format_ytick(t.get_text()) for t in ax.get_yticklabels()])

		_add_gaps(ax, list(pivot_df.columns))
		_bold_special_xticks(ax, list(pivot_df.columns))
		_decorate_cells(ax, list(pivot_df.columns), n_rows=len(pivot_df.index))
	plt.tight_layout()
	plt.savefig(outpath_pdf, dpi=250, bbox_inches="tight", pad_inches=0)
	if args.show_figures:
		plt.show()
	plt.close()


# -----------------------------
# Main Execution
# -----------------------------

def main():
	results_dir = args.results_dir
	outdir = Path(args.outdir)
	_ensure_outdir(outdir)

	df = load_all_results(results_dir)
	print(f"Loaded {len(df)} rows | models={df['model'].nunique()} | strategies={df['strategy'].nunique()} | biases={df['bias'].nunique()}.")

	# Expand tiers
	long_df = expand_complexity_tiers(df)

	# =============== STRATEGY-LEVEL SUMMARIES (AGGREGATED OVER MODELS) ===============
	# Overall means by strategy
	strat_overall = aggregate_strategy_overall(df)
	strat_overall.to_csv(outdir / "means_overall_by_strategy.csv", index=False)

	# Per-tier means by strategy
	if not long_df.empty:
		strat_tier = aggregate_strategy_by_tier(long_df)
		strat_tier.to_csv(outdir / "means_by_tier_by_strategy.csv", index=False)

	# Winners per bias (which strategy has highest sensitivity per bias after averaging over models)
	per_bias_avg = (df.dropna(subset=['sensitivity'])
					  .groupby(['bias','strategy'])['sensitivity']
					  .median()
					  .reset_index())
	per_bias_avg['rank_within_bias'] = per_bias_avg.groupby(['bias'])['sensitivity'].rank(ascending=True, method='min')
	winners = per_bias_avg[per_bias_avg['rank_within_bias'] == 1].sort_values(['bias'])
	winners.to_csv(outdir / "winners_by_bias_strategy.csv", index=False)

	# =============== OPTIONAL: MODEL-LEVEL BREAKOUTS (DISABLED BY DEFAULT) ===============
	if not args.skip_model_breakouts:
		# overall by model × strategy (for completeness)
		overall_mean = (df.dropna(subset=['sensitivity'])
						  .groupby(['model','strategy'])['sensitivity']
						  .agg(['mean','std','count'])
						  .reset_index()
						  .rename(columns={'mean':'mean_sensitivity','std':'sd','count':'n_biases'}))
		overall_mean.to_csv(outdir / "means_overall_by_model_strategy.csv", index=False)

		if not long_df.empty:
			tier_means = (long_df.dropna(subset=['tier_sensitivity'])
							.groupby(['model','strategy','tier'])['tier_sensitivity']
							.agg(['mean','std','count'])
							.reset_index()
							.rename(columns={'mean':'mean_sensitivity','std':'sd','count':'n_biases'}))
			tier_means.to_csv(outdir / "means_by_tier_by_model_strategy.csv", index=False)

	# =============== FIGURES (BOXPLOTS) ===============
	# Overall: distribution across biases for each strategy (values averaged over models per bias)
	fig_out = outdir / "fig_overall_by_strategy_boxplot.pdf"
	aggregated_samples_overall = per_bias_avg#.rename(columns={'sensitivity': 'value'})
	samples_overall = df.dropna(subset=['sensitivity'])#.rename(columns={'sensitivity': 'value'})
	plot_box_by_strategy(
		aggregated_samples_df=aggregated_samples_overall,
		samples_df=samples_overall,
		label_col='strategy',
		unit_col='bias',
		value_col='sensitivity',
		title='Sensitivity by strategy (distribution across biases × models)',
		outpath=fig_out,
		outpath_stats_csv=outdir / "stats_pairwise_overall_by_strategy.csv",
	)

	# --- Add "all biases" row ---
	all_biases_df = df.copy()
	all_biases_df['bias'] = "all biases"  # same col used for row_col

	# Append back to original df
	df_with_all = pd.concat([df, all_biases_df], ignore_index=True)
	fig_out_bias = outdir / "fig_overall_by_strategy_and_bias_heatmap.pdf"
	plot_heatmap(
		df=df_with_all,
		outpath_pdf=fig_out_bias,
		row_col="bias",
		col_col="strategy",
		value_col="sensitivity",
		samples_df=df_with_all,
		outpath_csv=outdir / "pivot_bias_vs_strategy.csv",
		outpath_stats_csv=outdir / "stats_bias_vs_strategy_long.csv",
		outpath_pvals_csv=outdir / "pvals_bias_vs_strategy.csv",
		outpath_qvals_csv=outdir / "qvals_bias_vs_strategy_fdr.csv",
		fontsize=DEFAULT_FONTSIZE,
		split_strategy_groups=True
	)

	# 3) Model × Strategy (the new “strategy vs model” view)
	fig_out_sxm = outdir / "fig_strategy_vs_model_heatmap.pdf"
	plot_heatmap(
		df=df_with_all,
		outpath_pdf=fig_out_sxm,
		row_col="model",
		col_col="strategy",
		value_col="sensitivity",
		samples_df=df_with_all,
		outpath_csv=outdir / "pivot_model_vs_strategy.csv",
		outpath_stats_csv=outdir / "stats_model_vs_strategy_long.csv",
		outpath_pvals_csv=outdir / "pvals_model_vs_strategy.csv",
		outpath_qvals_csv=outdir / "qvals_model_vs_strategy_fdr.csv",
		fontsize=DEFAULT_FONTSIZE,
		split_strategy_groups=True
	)

	# Per-tier: distribution across biases within each tier
	if not long_df.empty:

		aggregated_samples_all = (
			long_df.dropna(subset=['tier_sensitivity'])
				   .groupby(['bias','tier','strategy'], as_index=False)['tier_sensitivity']
				   .mean()
				   .rename(columns={'tier_sensitivity':'sensitivity'})
		)
		raw_samples_all = long_df.rename(columns={'tier_sensitivity':'sensitivity'})
		fig_out_tiers = outdir / "fig_by_strategy_ALL_TIERS_heatmap.pdf"
		plot_heatmap(
			df=aggregated_samples_all,
			outpath_pdf=fig_out_tiers,
			row_col="tier",
			col_col="strategy",
			value_col="sensitivity",
			samples_df=raw_samples_all,
			outpath_csv=outdir / "pivot_tier_vs_strategy.csv",
			outpath_stats_csv=outdir / "stats_tier_vs_strategy_long.csv",
			outpath_pvals_csv=outdir / "pvals_tier_vs_strategy.csv",
			outpath_qvals_csv=outdir / "qvals_tier_vs_strategy_fdr.csv",
			fontsize=DEFAULT_FONTSIZE,
			split_strategy_groups=True
		)

		for tier in (strat_tier['tier'].cat.categories if hasattr(strat_tier['tier'], 'cat') else sorted(strat_tier['tier'].unique())):
			fig_out_tier = outdir / f"fig_by_strategy_tier={tier}_boxplot.pdf"
			aggregated_samples_tier = (long_df.dropna(subset=['tier_sensitivity'])
							   .groupby(['bias','strategy','tier'])['tier_sensitivity']
							   .mean()
							   .reset_index()
							   .rename(columns={'tier_sensitivity':'sensitivity'}))
			aggregated_samples_tier = aggregated_samples_tier[aggregated_samples_tier['tier'] == tier]

			# Raw bias×model values within tier (with stats)
			raw_samples_tier = (
				long_df.dropna(subset=['tier_sensitivity'])
				.rename(columns={'tier_sensitivity': 'sensitivity'})
			)
			raw_samples_tier = raw_samples_tier[raw_samples_tier['tier'] == tier]


			stats_t = tier_stats[tier_stats['tier'] == tier] if 'tier_stats' in locals() else None
			plot_box_by_strategy(
				aggregated_samples_df=aggregated_samples_tier,
				samples_df=raw_samples_tier,
				label_col='strategy',
				unit_col='bias',
				value_col='sensitivity',
				title=f'Sensitivity by strategy – Complexity: {tier}',
				outpath=fig_out_tier,
				outpath_stats_csv=outdir / f"stats_pairwise_tier={tier}_by_strategy.csv",
			)


	print(f"\nWrote analysis tables and figures to: {outdir.resolve()}\n")
	print("Key files:")
	print(" - means_overall_by_strategy.csv")
	print(" - means_by_tier_by_strategy.csv")
	print(" - winners_by_bias_strategy.csv")
	print(" - stats_pairwise_overall_by_strategy.csv")
	print(" - stats_pairwise_by_tier_by_strategy.csv")
	print(" - fig_overall_mean_by_strategy.pdf")
	print(" - fig_mean_by_strategy_tier=*.pdf")

if __name__ == "__main__":
	main()
