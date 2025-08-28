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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
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
strategy_mapping = {
	"None": [],  # no strategy
	"AX": ["inject_axioms"], 
	"BW": ["bias_warning_in_system_instruction"], 
	"CoT": ["chain_of_thought"], 
	"IsD": ["impersonified_self_debiasing"], 
	"IMP": ["implication_prompting"], 
	"2sAX": ["two_step_axioms_elicitation"], 
	"1sAX": ["one_step_axioms_elicitation"], 
	"2sAX+BW": ["bias_warning_in_system_instruction", "two_step_axioms_elicitation"], 
	"1sAX+BW": ["bias_warning_in_system_instruction", "one_step_axioms_elicitation"],
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
				'bias': row['bias'], 			'model': row['model'], 			'strategy': row['strategy'], 			'tier': tier, 			'tier_total_cases': metrics.get('total_cases', np.nan), 			'tier_sensitivity': metrics.get('sensitivity', np.nan), 			'tier_harmfulness': metrics.get('harmfulness', np.nan), 			'tier_prolog_uncertainty': metrics.get('prolog_uncertainty', np.nan), 		})
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
		if s == "None":
			rbs_results[s] = {"p": "-", "rbs": "0"}
			continue

		df_s = samples_df[samples_df[label_col] == s][[unit_col, value_col]].dropna().rename(
			columns={value_col: "val_s"}
		)
		df_none = samples_df[samples_df[label_col] == "None"][[unit_col, value_col]].dropna().rename(
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
		if s == "None":
			return 0
		v = rbs_results.get(s, {}).get("rbs")
		return v if isinstance(v, (float, int)) and np.isfinite(v) else np.inf

	labels = sorted(unique_values, key=rbs_key)

	# --- Plot ---
	plt.figure(figsize=(10, 4))
	x = np.arange(len(labels))

	# prepare box colors: highlight "None"
	box_colors = ["lightgray" if s == "None" else "white" for s in labels]

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

	# --- Highlight the 'AX' tick (label + tick mark) ---
	if "AX" in labels:
		ax = plt.gca()
		idx_ax = labels.index("AX")

		# color the tick label
		ax.get_xticklabels()[idx_ax].set_color("tab:red")
		ax.get_xticklabels()[idx_ax].set_fontweight("bold")

		# color the tick mark lines too (optional)
		tick_ax = ax.xaxis.get_major_ticks()[idx_ax]
		tick_ax.tick1line.set_color("tab:red")
		tick_ax.tick2line.set_color("tab:red")

	if "None" in labels:
		ax = plt.gca()
		idx_ax = labels.index("None")

		# color the tick label
		ax.get_xticklabels()[idx_ax].set_color("tab:green")
		ax.get_xticklabels()[idx_ax].set_fontweight("bold")

		# color the tick mark lines too (optional)
		tick_ax = ax.xaxis.get_major_ticks()[idx_ax]
		tick_ax.tick1line.set_color("tab:green")
		tick_ax.tick2line.set_color("tab:green")

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

	# --- Mean and median annotations ---
	for i, s in enumerate(labels):
		m = means[s]
		if np.isfinite(m):
			plt.plot(x[i], m, marker="D", color="black")
			plt.text(x[i], m + y_offset_mean, f"{m:.2f}", ha="center", va="bottom", fontsize=fontsize,
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
			plt.text(x[i], med + y_offset_median, f"{med:.2f}", ha="center", va="bottom", fontsize=fontsize,
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
	plt.savefig(outpath, dpi=200)
	if args.show_figures:
		plt.show()
	plt.close()

def plot_heatmap(
	df,
	outpath,
	value_col="sensitivity",
	cmap="icefire",
	center=None,                # e.g. 0 for diverging metrics; otherwise None
	fontsize=12
):
	# --- Aggregate to Bias × Strategy pivot ---
	pivot_df = (
		df.dropna(subset=[value_col])
		  .groupby(["bias", "strategy"], as_index=False)[value_col]
		  .median()
		  .pivot(index="bias", columns="strategy", values=value_col)
	)

	if pivot_df.size == 0:
		raise ValueError("No data to plot after grouping/pivot. Check 'bias', 'strategy', and value_col.")

	# --- Ensure "None" is the last column ---
	if "None" in pivot_df.columns:
		pivot_df = pivot_df[[c for c in pivot_df.columns if c != "None"] + ["None"]]

	# --- Compute p-values (strategy vs None per bias) ---
	pval_dict = {}
	for bias in pivot_df.index:
		for strat in pivot_df.columns:
			if strat == "None":
				pval_dict[(bias, strat)] = (None, 0)
				continue
			vals_strat = df[(df["bias"] == bias) & (df["strategy"] == strat)][value_col].dropna()
			vals_none  = df[(df["bias"] == bias) & (df["strategy"] == "None")][value_col].dropna()
			if len(vals_strat) > 0 and len(vals_none) > 0:
				try:
					stat, p = mannwhitneyu(vals_strat, vals_none, alternative="less")
					rbs = compute_rbs(vals_strat, vals_none, stat)
					pval_dict[(bias, strat)] = (p, rbs)
				except Exception:
					pval_dict[(bias, strat)] = (None, 0)
			else:
				pval_dict[(bias, strat)] = (None, 0)

	# --- EFFECT-BASED 'best' mask: pick largest |r_rb| per bias (excluding "None") ---
	# Build an effect-size DataFrame aligned to pivot_df
	effect_df = pivot_df.copy().astype(float)
	effect_df.loc[:, :] = float("nan")
	for bias in pivot_df.index:
		for strat in pivot_df.columns:
			if strat == "None":
				continue
			p, rbs = pval_dict.get((bias, strat), (None, float("nan")))
			effect_df.loc[bias, strat] = (rbs if p is not None else float("nan"))

	# for rows that have at least one non-NaN effect, override fallback with effect-based best
	effect_has_values = effect_df.notna().any(axis=1)
	if effect_has_values.any():
		# per-row maximum absolute effect
		row_max = effect_df.max(axis=1, skipna=True)
		best_mask = effect_df.eq(row_max, axis=0)
		# never select "None" as best
		if "None" in best_mask.columns:
			best_mask["None"] = False

	# --- Tie-break: if multiple True in a row, keep the one with the lowest metric ---
	# Uses the numeric values from pivot_df (underlying metric behind ann_df).
	for bias in pivot_df.index:
		row_mask = best_mask.loc[bias]
		if row_mask.sum() > 1:
			candidates = row_mask[row_mask].index
			# choose the column with the minimal metric value (ignoring NaNs)
			chosen = pivot_df.loc[bias, candidates].astype(float).idxmin()
			best_mask.loc[bias, :] = False
			best_mask.loc[bias, chosen] = True

	# --- Plot ---
	plt.figure(figsize=(len(pivot_df.columns) * 1.25, max(2.5, len(pivot_df) * 0.5)))
	ann_df = pivot_df.copy().astype(float)
	for i, bias in enumerate(pivot_df.index):
		for j, strat in enumerate(pivot_df.columns):
			val = ann_df.loc[bias, strat]
			p = pval_dict[(bias, strat)][0]
			stars = (
				r"$^{***}$" if p < 0.001 else
				(r"$^{**}$" if p < 0.01 else
				 (r"$^{*}$" if p < 0.05 else ""))
			) if p else ''
			ann_df.loc[bias, strat] = f"{val:.1f}%{stars}"

	ax = sns.heatmap(
		pivot_df,
		annot=ann_df,
		fmt="",
		cmap=cmap,
		center=center,
		cbar_kws={"label": value_col.capitalize(), "shrink": 0.9, "pad": 0.01},
		linewidths=0.5,
		linecolor="gray"
	)
	plt.ylabel("")
	plt.xlabel("")

	n_rows, n_cols = pivot_df.shape

	# --- Small whitespace gap before "None" instead of a thick line ---
	if "None" in pivot_df.columns:
		none_idx = pivot_df.columns.get_loc("None")  # left edge of "None"
		gap = 0.12  # width of the gap in "cell" units; try 0.08–0.2
		bg = ax.figure.get_facecolor()  # match the figure background
		# Center a narrow band on the boundary (won't cover text centered in cells)
		ax.axvspan(none_idx - gap/2, none_idx + gap/2,
				   color=bg, zorder=6, lw=0)

	ann = {(i, j): ax.texts[i * n_cols + j] for i in range(n_rows) for j in range(n_cols)}

	# --- Highlight best cells & add p-values ---
	for i, bias in enumerate(pivot_df.index):
		for j, strat in enumerate(pivot_df.columns):
			significant_pvalue = (pval_dict[(bias, strat)][0] and pval_dict[(bias, strat)][0] < 0.05)
			if best_mask.iat[i, j]:
				ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))
			if significant_pvalue:
				t = ann[(i, j)]
				t.set_fontweight("bold")
				t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])
				t.set_color("yellow")
			# Add p-value text
			p_text = fr"$r_{{rb}}={pval_dict[(bias, strat)][-1]:.2f}$" if abs(pval_dict[(bias, strat)][-1]) >= 0.01 else ""
			if p_text not in ["", "-"]:
				ax.text(
					j + 0.5, i + 0.67, p_text,
					ha="center", va="top",
					fontstyle="italic",
					fontweight=ax.texts[i * n_cols + j].get_fontweight(),
					fontsize=fontsize-5,
					color=ax.texts[i * n_cols + j].get_color(),
					path_effects=ax.texts[i * n_cols + j].get_path_effects(),
				)

	# --- Highlight the 'AX' tick (label + tick mark) ---
	if "AX" in pivot_df.columns:
		idx_ax = pivot_df.columns.get_loc("AX")

		# color the tick label
		ax.get_xticklabels()[idx_ax].set_color("tab:red")
		ax.get_xticklabels()[idx_ax].set_fontweight("bold")

		# color the tick mark lines too (optional)
		tick_ax = ax.xaxis.get_major_ticks()[idx_ax]
		tick_ax.tick1line.set_color("tab:red")
		tick_ax.tick2line.set_color("tab:red")

	if "None" in pivot_df.columns:
		idx_ax = pivot_df.columns.get_loc("None")

		# color the tick label
		ax.get_xticklabels()[idx_ax].set_color("tab:green")
		ax.get_xticklabels()[idx_ax].set_fontweight("bold")

		# color the tick mark lines too (optional)
		tick_ax = ax.xaxis.get_major_ticks()[idx_ax]
		tick_ax.tick1line.set_color("tab:green")
		tick_ax.tick2line.set_color("tab:green")

	plt.tight_layout()
	plt.savefig(outpath, dpi=250)
	if args.show_figures:
		plt.show()
	plt.close()

def plot_heatmap_by_strategy_tier_best(
	aggregated_samples_df,
	samples_df,
	outpath,
	tier_col="tier",
	label_col="strategy",
	value_col="sensitivity",
	cmap="icefire",
	center=None,                # e.g., 0 for diverging metrics
	fontsize=12
):
	# --- Pivot to Tier × Strategy ---
	pivot_df = (
		aggregated_samples_df
		.dropna(subset=[value_col])
		.groupby([tier_col, label_col], as_index=False)[value_col]
		.median()
		.pivot(index=tier_col, columns=label_col, values=value_col)
	)

	if pivot_df.size == 0:
		raise ValueError("No data to plot after grouping/pivot.")

	# --- Ensure "None" is the last column ---
	if "None" in pivot_df.columns:
		pivot_df = pivot_df[[c for c in pivot_df.columns if c != "None"] + ["None"]]

	# --- Compute p-values (strategy vs None per tier) ---
	pval_dict = {}
	for tier in pivot_df.index:
		for strat in pivot_df.columns:
			if strat == "None":
				pval_dict[(tier, strat)] = (None, 0)
				continue
			vals_strat = samples_df[(samples_df[tier_col] == tier) & (samples_df[label_col] == strat)][value_col].dropna()
			vals_none = samples_df[(samples_df[tier_col] == tier) & (samples_df[label_col] == "None")][value_col].dropna()
			if len(vals_strat) > 0 and len(vals_none) > 0:
				try:
					stat, p = mannwhitneyu(vals_strat, vals_none, alternative="less")
					rbs = compute_rbs(vals_strat, vals_none, stat)
					pval_dict[(tier, strat)] = (p, rbs)
				except Exception:
					pval_dict[(tier, strat)] = (None, 0)
			else:
				pval_dict[(tier, strat)] = (None, 0)

	# --- EFFECT-BASED 'best' mask: pick largest |r_rb| per bias (excluding "None") ---
	# Build an effect-size DataFrame aligned to pivot_df
	effect_df = pivot_df.copy().astype(float)
	effect_df.loc[:, :] = float("nan")
	for bias in pivot_df.index:
		for strat in pivot_df.columns:
			if strat == "None":
				continue
			p, rbs = pval_dict.get((bias, strat), (None, float("nan")))
			effect_df.loc[bias, strat] = (rbs if p is not None else float("nan"))

	# for rows that have at least one non-NaN effect, override fallback with effect-based best
	effect_has_values = effect_df.notna().any(axis=1)
	if effect_has_values.any():
		# per-row maximum absolute effect
		row_max = effect_df.max(axis=1, skipna=True)
		best_mask = effect_df.eq(row_max, axis=0)
		# never select "None" as best
		if "None" in best_mask.columns:
			best_mask["None"] = False

	# --- Tie-break: if multiple True in a row, keep the one with the lowest metric ---
	# Uses the numeric values from pivot_df (underlying metric behind ann_df).
	for bias in pivot_df.index:
		row_mask = best_mask.loc[bias]
		if row_mask.sum() > 1:
			candidates = row_mask[row_mask].index
			# choose the column with the minimal metric value (ignoring NaNs)
			chosen = pivot_df.loc[bias, candidates].astype(float).idxmin()
			best_mask.loc[bias, :] = False
			best_mask.loc[bias, chosen] = True

	# --- Plot ---
	plt.figure(figsize=(len(pivot_df.columns) * 1.15, max(2.5, len(pivot_df) * 0.5)))
	# ann_df  = pivot_df.applymap(lambda v: f"{v:.1f}% (r={effect_size:.1f})")
	ann_df = pivot_df.copy().astype(float)
	for i, tier in enumerate(pivot_df.index):
		for j, strat in enumerate(pivot_df.columns):
			val = ann_df.loc[tier, strat]
			p = pval_dict[(tier, strat)][0]
			stars = (
				r"$^{***}$" if p < 0.001 else
				(r"$^{**}$" if p < 0.01 else
				 (r"$^{*}$" if p < 0.05 else ""))
			) if p else ''
			ann_df.loc[tier, strat] = f"{val:.1f}%{stars}"
	ax = sns.heatmap(
		pivot_df,
		annot=ann_df,
		fmt="",
		cmap=cmap,
		center=center,
		cbar_kws={"label": value_col.capitalize(), "shrink": 0.9, "pad": 0.01},
		linewidths=0.5,
		linecolor="gray"
	)
	plt.ylabel("")
	plt.xlabel("")

	n_rows, n_cols = pivot_df.shape

	# --- Small whitespace gap before "None" instead of a thick line ---
	if "None" in pivot_df.columns:
		none_idx = pivot_df.columns.get_loc("None")  # left edge of "None"
		gap = 0.12  # width of the gap in "cell" units; try 0.08–0.2
		bg = ax.figure.get_facecolor()  # match the figure background
		# Center a narrow band on the boundary (won't cover text centered in cells)
		ax.axvspan(none_idx - gap/2, none_idx + gap/2,
				   color=bg, zorder=6, lw=0)

	ann = {(i, j): ax.texts[i * n_cols + j] for i in range(n_rows) for j in range(n_cols)}

	# --- Highlight best cells & add p-values ---
	for i, tier in enumerate(pivot_df.index):
		for j, strat in enumerate(pivot_df.columns):
			significant_pvalue = (pval_dict[(tier, strat)][0] and pval_dict[(tier, strat)][0] < 0.05)
			if best_mask.iat[i, j]:
				# rectangle
				ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=2))
			if significant_pvalue:
				# bold white text
				t = ann[(i, j)]
				t.set_fontweight("bold")
				t.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])
				t.set_color("yellow")
			# Add p-value text
			p_text = fr"$r_{{rb}}={pval_dict[(tier, strat)][-1]:.2f}$" if abs(pval_dict[(tier, strat)][-1]) >= 0.01 else ""
			if p_text not in ["", "-"]:
				ax.text(
					j + 0.5, i + 0.67, p_text,
					ha="center", va="top",
					fontstyle="italic",
					fontweight=ax.texts[i * n_cols + j].get_fontweight(),
					fontsize=fontsize-2,
					color=ax.texts[i * n_cols + j].get_color(),
					path_effects=ax.texts[i * n_cols + j].get_path_effects(),
					# bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2")
				)

	# --- Highlight the 'AX' tick (label + tick mark) ---
	if "AX" in pivot_df.columns:
		idx_ax = pivot_df.columns.get_loc("AX")

		# color the tick label
		ax.get_xticklabels()[idx_ax].set_color("tab:red")
		ax.get_xticklabels()[idx_ax].set_fontweight("bold")

		# color the tick mark lines too (optional)
		tick_ax = ax.xaxis.get_major_ticks()[idx_ax]
		tick_ax.tick1line.set_color("tab:red")
		tick_ax.tick2line.set_color("tab:red")

	if "None" in pivot_df.columns:
		idx_ax = pivot_df.columns.get_loc("None")

		# color the tick label
		ax.get_xticklabels()[idx_ax].set_color("tab:green")
		ax.get_xticklabels()[idx_ax].set_fontweight("bold")

		# color the tick mark lines too (optional)
		tick_ax = ax.xaxis.get_major_ticks()[idx_ax]
		tick_ax.tick1line.set_color("tab:green")
		tick_ax.tick2line.set_color("tab:green")

	plt.tight_layout()
	plt.savefig(outpath, dpi=250)
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

	fig_out_heat = outdir / "fig_overall_by_strategy_and_bias_heatmap.pdf"
	plot_heatmap(samples_overall, fig_out_heat, cmap="icefire", value_col="sensitivity")

	# Per-tier: distribution across biases within each tier
	if not long_df.empty:
		aggregated_samples_all = (
			long_df.dropna(subset=['tier_sensitivity'])
				   .groupby(['bias','strategy','tier'])['tier_sensitivity']
				   .mean()
				   .reset_index()
				   .rename(columns={'tier_sensitivity':'sensitivity'})
		)

		raw_samples_all = (
			long_df.dropna(subset=['tier_sensitivity'])
				   .rename(columns={'tier_sensitivity': 'sensitivity'})
		)

		fig_out_all_hm = outdir / "fig_by_strategy_ALL_TIERS_heatmap.pdf"

		plot_heatmap_by_strategy_tier_best(
			aggregated_samples_all,
			raw_samples_all,
			fig_out_all_hm,
			tier_col="tier",
			label_col="strategy",
			value_col="sensitivity",
			cmap="coolwarm",
			center=None,                # e.g., 0 for diverging metrics
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
