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

def plot_box_by_strategy(aggregated_samples_df, samples_df, label_col, unit_col, value_col, title, outpath, fontsize=DEFAULT_FONTSIZE):

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

	# --- Mann-Whitney U + rank biserial effect size ---
	rbs_results = {}
	for s in unique_values:
		if s == NO_STRATEGY_LABEL:
			rbs_results[s] = {"p": "-", "rbs": "0"}
			continue

		df_s = samples_df[samples_df[label_col] == s][[unit_col, value_col]].dropna().rename(
			columns={value_col: "val_s"}
		)
		df_none = samples_df[samples_df[label_col] == NO_STRATEGY_LABEL][[unit_col, value_col]].dropna().rename(
			columns={value_col: "val_none"}
		)
		merged = pd.merge(df_s, df_none, on=unit_col, how="inner").sort_values(by=unit_col)

		_x = merged["val_s"].to_numpy()
		_y = merged["val_none"].to_numpy()

		if len(_x) == 0 or len(_y) == 0:
			rbs_results[s] = {"p": "n/a", "rbs": "n/a"}
			continue

		try:
			stat, p = mannwhitneyu(_x, _y, alternative="less")
			rbs = compute_rbs(_x, _y, stat)
			rbs_results[s] = {"p": p, "rbs": rbs}
		except ValueError:
			rbs_results[s] = {"p": "err", "rbs": "err"}

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

	# special tick colors
	ax = plt.gca()
	for c in labels:
		if c in [PROBE_AXIOMS_INJECTION_LABEL,NO_STRATEGY_LABEL]:
			idx = labels.index(c)
			ax.get_xticklabels()[idx].set_color("tab:red")
			ax.get_xticklabels()[idx].set_fontweight("bold")
			t = ax.xaxis.get_major_ticks()[idx]
			t.tick1line.set_color("tab:red"); t.tick2line.set_color("tab:red")
		elif 'AX' in c:
			idx = labels.index(c)
			ax.get_xticklabels()[idx].set_color("tab:green")
			ax.get_xticklabels()[idx].set_fontweight("bold")
			t = ax.xaxis.get_major_ticks()[idx]
			t.tick1line.set_color("tab:green"); t.tick2line.set_color("tab:green")

	# --- Compute axis limits and spacing ---
	finite_maxes = [np.nanmax(v) for v in data.values() if v.size]
	finite_mins = [np.nanmin(v) for v in data.values() if v.size]
	data_max = np.nanmax(finite_maxes) if finite_maxes else 1.0
	data_min = np.nanmin(finite_mins) if finite_mins else 0.0
	y_range = (data_max - data_min) if np.isfinite(data_max) and np.isfinite(data_min) else 1.0
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

	# --- Annotate p-values and RBS ---
	for i, s in enumerate(labels):
		stats_result = rbs_results.get(s, {})
		p = stats_result.get("p", "")
		rbs = stats_result.get("rbs", "")
		if isinstance(p, float):
			p_text = f"p={p:.3f}"
			if p < 0.001:
				p_text = f"p<0.001"
			else:
				p_text = f"p={p:.3f}"
			rbs_text = fr"$\bf{{r_{{rb}}\!=\!{rbs:.2f}}}$" if p < 0.05 else fr"$r_{{rb}}={rbs:.2f}$"
			fontweight = "bold" if p < 0.05 else "normal"
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
	plt.savefig(outpath, dpi=200, bbox_inches="tight")
	if args.show_figures:
		plt.show()
	plt.close()

def plot_heatmap(
	df,
	outpath_pdf,
	*,
	row_col,                     # e.g. "bias", "tier", or "model"
	col_col="strategy",         # usually "strategy"
	value_col="sensitivity",
	samples_df=None,            # raw samples for stats; if None, skip p-values/effect sizes
	baseline_col_value=NO_STRATEGY_LABEL,   # Ø baseline for Mann–Whitney
	exclude_from_best=None,     # defaults to [Ø, ProbeAX]; pass [] to include all
	pre_agg_cols=None,          # optional: group these cols first (e.g. ["bias","tier","strategy"])
	pre_agg_func="mean",        # used with pre_agg_cols, before pivot; then pivot does median
	outpath_csv=None,
	cmap="viridis_r",
	center=None,
	fontsize=12,
	gap_width=0.12,             # narrow whitespace around ProbeAX/Ø
	show=False                  # mirror args.show_figures behavior if you want
):
	"""
	Universal heatmap:
	  - Pivots df by (row_col × col_col) with values = median(value_col) after optional pre-aggregation.
	  - Orders strategies like `strategy_mapping`, puts ProbeAX first and Ø last (if present).
	  - Per-row significance: Mann–Whitney U (alternative='less') vs baseline_col_value within each row (if samples_df provided).
	  - Annotates each cell with value% + stars, and italic r_rb below; bolds the best strategy per row
		(primary = min metric, tie-break = max |r_rb|), excluding Ø and ProbeAX by default.
	  - Applies row-wise z-normalization for coloring, shows labels as actual %.
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

	# ---------- per-row p-values & rank-biserial effect sizes ----------
	pval_dict = {}
	if samples_df is not None:
		for r in pivot_df.index:
			for c in pivot_df.columns:
				if c == baseline_col_value:
					pval_dict[(r, c)] = (None, 0)
					continue
				vals_s = samples_df[(samples_df[row_col] == r) & (samples_df[col_col] == c)][value_col].dropna()
				vals_0 = samples_df[(samples_df[row_col] == r) & (samples_df[col_col] == baseline_col_value)][value_col].dropna()
				if len(vals_s) > 0 and len(vals_0) > 0:
					try:
						stat, p = mannwhitneyu(vals_s, vals_0, alternative="less")
						rbs = compute_rbs(vals_s, vals_0, stat)
						pval_dict[(r, c)] = (p, rbs)
					except Exception:
						pval_dict[(r, c)] = (None, 0)
				else:
					pval_dict[(r, c)] = (None, 0)

	# ---------- best-cell mask (min metric; tie-break max |r_rb|) ----------
	if exclude_from_best is None:
		exclude_from_best = [baseline_col_value, PROBE_AXIOMS_INJECTION_LABEL]
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

	# ---------- annotations (value% + stars) ----------
	ann_df = pivot_df.copy().astype(float)
	for i, r in enumerate(pivot_df.index):
		for j, c in enumerate(pivot_df.columns):
			v = ann_df.loc[r, c]
			if samples_df is not None:
				p = pval_dict.get((r, c), (None,))[0]
				stars = (r"$^{***}$" if p and p < 0.001 else
						 (r"$^{**}$" if p and p < 0.01 else
						  (r"$^{*}$" if p and p < 0.05 else "")))
			else:
				stars = ""
			ann_df.loc[r, c] = f"{v:.1f}%{stars}"

	# ---------- color by row-normalized z, label with actual % ----------
	z = (pivot_df - pivot_df.mean(axis=1).values[:, None]) / pivot_df.std(axis=1).replace(0, np.nan).values[:, None]
	plt.figure(figsize=(len(pivot_df.columns) * .9, max(2.8, len(pivot_df) * 0.55)))
	ax = sns.heatmap(
		z,
		annot=ann_df,
		fmt="",
		vmin=np.nanpercentile(z.values, 20),
		vmax=np.nanpercentile(z.values, 80),
		cmap=cmap,
		center=center,
		cbar_kws={"label": "Row-norm z", "pad": 0.01, "aspect": 30},
		linewidths=0.5,
		linecolor="gray"
	)
	ax.set_ylabel("")
	ax.set_xlabel("")
	plt.xticks(rotation=25, ha="right")

	# Replace spaces in xtick labels with newlines
	new_labels = [lbl.get_text().replace(" ", "\n") for lbl in ax.get_yticklabels()]
	ax.set_yticklabels(new_labels)

	# subtle gaps around Ø and after ProbeAX
	if baseline_col_value in pivot_df.columns:
		idx = pivot_df.columns.get_loc(baseline_col_value)
		ax.axvspan(idx - gap_width/2, idx + gap_width/2, color=ax.figure.get_facecolor(), zorder=6, lw=1)
	if PROBE_AXIOMS_INJECTION_LABEL in pivot_df.columns:
		idx = pivot_df.columns.get_loc(PROBE_AXIOMS_INJECTION_LABEL) + 1
		ax.axvspan(idx - gap_width/2, idx + gap_width/2, color=ax.figure.get_facecolor(), zorder=6, lw=1)

	n_rows, n_cols = pivot_df.shape
	ann = {(i, j): ax.texts[i * n_cols + j] for i in range(n_rows) for j in range(n_cols)}

	# --- Highlight best cells & add p-values ---
	for i, r in enumerate(pivot_df.index):
		for j, c in enumerate(pivot_df.columns):
			significant_pvalue = (pval_dict[(r, c)][0] and pval_dict[(r, c)][0] < 0.05)
			if best_mask.iat[i, j]:
				t = ann[(i, j)]
				t.set_fontweight("bold")
				# t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])
				# t.set_color("yellow")
			# if best_mask.iat[i, j]:
			# 	ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="magenta", lw=2))
			# if significant_pvalue:
			# 	t = ann[(i, j)]
			# 	t.set_fontweight("bold")
			# 	t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])
			# 	t.set_color("yellow")
			# Add p-value text
			if samples_df is not None:
				p, effect_size = pval_dict[(r, c)]
				p_text = (fr"$r_{{rb}}\!=\!{effect_size:.2f}$" if abs(effect_size) >= 0.01 else fr"$|r_{{rb}}|\!<\!0.01$") if p else ""
				if p_text not in ["", "-"]:
					ax.text(
						j + 0.5, i + 0.67, p_text,
						ha="center", va="top",
						fontstyle="italic",
						fontweight=ax.texts[i * n_cols + j].get_fontweight(),
						fontsize=fontsize-2,
						color=ax.texts[i * n_cols + j].get_color(),
						path_effects=ax.texts[i * n_cols + j].get_path_effects(),
					)

	# special tick colors
	for c in pivot_df.columns:
		if c in [PROBE_AXIOMS_INJECTION_LABEL,baseline_col_value]:
			idx = pivot_df.columns.get_loc(c)
			ax.get_xticklabels()[idx].set_color("tab:red")
			ax.get_xticklabels()[idx].set_fontweight("bold")
			t = ax.xaxis.get_major_ticks()[idx]
			t.tick1line.set_color("tab:red"); t.tick2line.set_color("tab:red")
		elif 'AX' in c:
			idx = pivot_df.columns.get_loc(c)
			ax.get_xticklabels()[idx].set_color("tab:green")
			ax.get_xticklabels()[idx].set_fontweight("bold")
			t = ax.xaxis.get_major_ticks()[idx]
			t.tick1line.set_color("tab:green"); t.tick2line.set_color("tab:green")

	plt.tight_layout()
	plt.savefig(outpath_pdf, dpi=250, bbox_inches="tight")
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
		unit_col='bias',   # still match by bias for alignment
		value_col='sensitivity',
		title='Sensitivity by strategy (distribution across biases × models)',
		outpath=fig_out
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
		samples_df=df_with_all,  # for p-values/effect sizes
		outpath_csv=outdir / "pivot_bias_vs_strategy.csv",
		fontsize=DEFAULT_FONTSIZE
	)

	# 3) Model × Strategy (the new “strategy vs model” view)
	fig_out_sxm = outdir / "fig_strategy_vs_model_heatmap.pdf"
	plot_heatmap(
		df=df,
		outpath_pdf=fig_out_sxm,
		row_col="model",
		col_col="strategy",
		value_col="sensitivity",
		samples_df=df,
		outpath_csv=outdir / "pivot_model_vs_strategy.csv",
		fontsize=DEFAULT_FONTSIZE
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
			pre_agg_cols=None,   # already pre-aggregated above
			outpath_csv=outdir / "pivot_tier_vs_strategy.csv",
			fontsize=DEFAULT_FONTSIZE
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
			plot_box_by_strategy(aggregated_samples_df=aggregated_samples_tier, samples_df=raw_samples_tier, label_col='strategy', unit_col='bias', value_col='sensitivity', title=f'Sensitivity by strategy – Complexity: {tier}', outpath=fig_out_tier)


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
