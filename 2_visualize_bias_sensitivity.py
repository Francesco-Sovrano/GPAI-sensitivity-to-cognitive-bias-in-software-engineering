import os
import json
import re
import glob
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import numpy as np
import math
import seaborn as sns
from collections import defaultdict
from scipy import stats

# ---- Force vector text embedding ----
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams["svg.fonttype"] = "none"

DEFAULT_FONTSIZE = 9

# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Provide reasoning, and format")
parser.add_argument(
	"--self_assessment",
	action="store_true",
)
parser.add_argument(
	"--seed_corpus_only",
	action="store_true",
	help="Restrict evaluation to only the human-seeded dilemmas (filtering out AI-generated ones)"
)
parser.add_argument(
	"--show_figures",
	action="store_true",
	help="Flag to control whether to show figures (default: False)"
)
parser.add_argument(
	"--temperature",
	type=float,
	default=0,
)
parser.add_argument(
	"--top_p",
	type=float,
	default=0,
)
parser.add_argument(
	"--sort_biases", 
	choices=["original","alpha"], 
	default="alpha", 
	help="Ordering of bias facets: original (file order), alpha (alphabetical), sensitivity (descending mean sensitivity across models)"
)

#######################################
### Prompt Engineering Strategies
parser.add_argument(
	"--inject_axioms_in_prolog",
	action="store_true",
	help=(
		"Embed the raw Prolog-encoded axioms into prompts as reasoning cues "
		"instead of the human-readable description"
	)
)
parser.add_argument(
	"--inject_axioms",
	action="store_true",
	help=(
		"Include each dilemma’s axioms_description text as reasoning cues "
		"when constructing the biased/unbiased prompts"
	)
)
parser.add_argument(
	"--bias_warning",
	action="store_true",
)
parser.add_argument(
	"--chain_of_thought",
	action="store_true",
)
parser.add_argument(
	"--impersonified_self_debiasing",
	action="store_true",
)
parser.add_argument(
	"--implication_prompting",
	action="store_true",
)
parser.add_argument(
	"--bistep_axioms_elicitation",
	action="store_true",
)
parser.add_argument(
	"--prolog_bistep_axioms_elicitation",
	action="store_true",
)
parser.add_argument(
	"--self_axioms_elicitation",
	action="store_true",
)
#######################################
args = parser.parse_args()

results_dir = './generated_output_data/'

# -----------------------------
# Model Name Mapping & Ordering
# -----------------------------
model_mapping = {
	"gpt-4.1-nano": "gpt-4.1-nano",
	"gpt-4.1-mini": "gpt-4.1-mini",
	"gpt-4o-mini": "gpt-4o-mini",
	"llama-3.1-8b-instant": "llama-3.1",
	"llama-3.3-70b-versatile": "llama-3.3",
	"deepseek-r1-distill-llama-70b": "deepseek-r1",
}

ordered_models = ["llama-3.1", "llama-3.3", "deepseek-r1", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o-mini"]

# -----------------------------
# Data Loading
# -----------------------------

def load_bias_json(json_file_path):
	with open(json_file_path, 'r') as f:
		data = json.load(f)

	# Convert top-level dict into a DataFrame
	df = pd.DataFrame.from_dict(data, orient='index').reset_index()
	df = df.rename(columns={'index': 'bias'})

	# Preserve the raw nested complexity analysis for later expansion
	if 'complexity_analysis' in df.columns:
		pass

	return df

def load_all_results(results_dir):
	base = Path(results_dir)
	df_list = []
	for data_model in model_mapping.keys():
		base_str = f"1_bias_sensitivity_model={data_model}"
		if not args.self_assessment:
			base_str += ".data_model_list=[[]*[]]"
		if args.seed_corpus_only:
			base_str += f".seed_corpus_only=True"
			
		if args.inject_axioms_in_prolog:
			base_str += f".inject_axioms_in_prolog=True"
		if args.inject_axioms:
			base_str += f".inject_axioms=True"

		if args.bias_warning:
			base_str += f".bias_warning=True"
		if args.chain_of_thought:
			base_str += f".chain_of_thought=True"
		if args.impersonified_self_debiasing:
			base_str += f".impersonified_self_debiasing=True"
		if args.implication_prompting:
			base_str += f".implication_prompting=True"
		if args.bistep_axioms_elicitation:
			base_str += f".bistep_axioms_elicitation=True"
		if args.prolog_bistep_axioms_elicitation:
			base_str += f".prolog_bistep_axioms_elicitation=True"
		if args.self_axioms_elicitation:
			base_str += f".self_axioms_elicitation=True"

		if args.temperature:
			pattern = base_str + f".temperature={args.temperature}.top_p={args.top_p}.json"
		else:
			pattern = base_str + f".json"
		path_list = list(map(lambda p: p.name, base.glob(pattern)))
		print(path_list)
		if not path_list:
			continue
		# Choose the longest (often latest) filename
		json_file = os.path.join(results_dir, sorted(path_list, key=len, reverse=True)[0])
		df = load_bias_json(json_file)
		df["model"] = data_model
		df_list.append(df)

	if df_list:
		return pd.concat(df_list, ignore_index=True)
	else:
		raise FileNotFoundError("No JSON files found in the results directory.")

# -----------------------------
# Utility Helpers
# -----------------------------

def wrap_model_labels(labels):
	"""Insert line break to wrap long model names."""
	return [label.replace("-", "\n") for label in labels]

# -----------------------------
# Primary Bias Sensitivity Plot
# -----------------------------

def plot_bias_sensitivities(df, output_path="2_bias_sensitivity_plots.png"):
	# Normalize model names
	df["model"] = df["model"].map(model_mapping).fillna(df["model"])
	df = df[df["model"].isin(ordered_models)]

	# Sort and filter models actually present
	present_models = sorted(df["model"].unique(), key=ordered_models.index)
	df["model"] = pd.Categorical(df["model"], categories=present_models, ordered=True)

	# Apply wrapped labels for display
	wrapped_labels = dict(zip(present_models, wrap_model_labels(present_models)))
	df["wrapped_model"] = df["model"].map(wrapped_labels)

	# Color palette (color-blind friendly)
	palette = sns.color_palette("colorblind", n_colors=len(present_models))
	# color_map = dict(zip(df["wrapped_model"].unique(), palette))
	color_map = dict(zip(df["model"].unique(), palette))

	# Determine bias ordering
	if args.sort_biases == 'alpha':
		biases = sorted(df['bias'].unique())
	else:  # original appearance order
		# preserve first-seen order
		seen = []
		for b in df['bias'].tolist():
			if b not in seen:
				seen.append(b)
		biases = seen

	# Subplot grid
	ncols = 4
	nrows = -(-len(biases) // ncols)

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 2 * nrows), constrained_layout=True)
	axes = axes.reshape(nrows, ncols)

	i=0
	for ax, bias in zip(axes.flat, biases):
		bias_df = df[df["bias"] == bias]
		sns.barplot(
			data=bias_df,
			# x="wrapped_model",
			x="model",
			y="sensitivity",
			ax=ax,
			order=present_models,#wrap_model_labels(present_models),
			palette=color_map
		)
		print('@'*10)
		print(bias, np.mean(bias_df['sensitivity']))
		ax.set_title(bias.capitalize(), fontsize=DEFAULT_FONTSIZE+2)
		if i%4 == 0:
			ax.set_ylabel('Sensitivity (%)')
		else:
			ax.set_ylabel('')
		ax.set_xlabel("")
		ax.set_ylim(0, 100)
		ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
		if i >= 4:
			ax.tick_params(axis='x', rotation=30)
			plt.setp(ax.get_xticklabels(), ha='right')
		else:
			ax.set_xticks([])  # Hides x-axis ticks

		# Annotate bars
		for bar in ax.patches:
			height = bar.get_height()
			if pd.notna(height):
				ax.text(
					bar.get_x() + bar.get_width() / 2,
					height + 2,
					f"{height:.1f}%",
					ha='center',
					va='bottom',
					fontsize=DEFAULT_FONTSIZE,
					bbox=dict(
						boxstyle="round,pad=0.2",
						facecolor="white",
						edgecolor="none",
						alpha=0.9
					)
				)
		i+=1

	# Remove unused subplots
	for i in range(len(biases), nrows * ncols):
		fig.delaxes(axes.flat[i])

	# fig.suptitle("Model Sensitivity by Bias", fontsize=16)
	plt.tight_layout()
	plt.savefig(output_path, dpi=300)
	if args.show_figures:
		plt.show()
	plt.close(fig)

# -----------------------------
# Complexity Tier Expansion & Plotting
# -----------------------------

def expand_complexity_tiers(df):
	"""
	Expand the nested complexity_analysis column into a long-form DataFrame with:
	bias, model, tier, sensitivity, harmfulness, prolog_uncertainty, total_cases.

	The JSON may have missing tiers for some biases/models; we handle that gracefully.
	"""
	records = []
	for _, row in df.iterrows():
		bias = row['bias']
		model = row['model']
		comp = row.get('complexity_analysis', {}) or {}
		if isinstance(comp, dict):
			for tier, metrics in comp.items():
				if not isinstance(metrics, dict):
					continue
				records.append({
					'bias': bias,
					'model': model,
					'tier': tier,
					'tier_total_cases': metrics.get('total_cases', np.nan),
					'tier_sensitivity': metrics.get('sensitivity', np.nan),
					'tier_harmfulness': metrics.get('harmfulness', np.nan),
					'tier_prolog_uncertainty': metrics.get('prolog_uncertainty', np.nan),
				})
	if not records:
		return pd.DataFrame(columns=['bias','model','tier','tier_total_cases','tier_sensitivity','tier_harmfulness','tier_prolog_uncertainty'])
	long_df = pd.DataFrame.from_records(records)

	# Normalize tier ordering (quartile-based). We accept synonyms / missing tiers.
	tier_order = ['low', 'mid-low', 'mid-high', 'high']
	long_df['tier'] = pd.Categorical(long_df['tier'], categories=tier_order, ordered=True)

	return long_df

def plot_complexity_sensitivity(long_df, overall_df, quartiles_df=None, output_path_prefix="2_bias_complexity_sensitivity"):
	"""
	Faceted line plots of sensitivity across complexity tiers with per-model baselines.
	Each facet includes a merged legend: model color + slope + final tier % + delta from baseline.
	"""
	if long_df.empty:
		print("No complexity_analysis data found; skipping complexity plots.")
		return

	# ---------------- Configuration ----------------
	SHOW_PER_PANEL_MODEL_LEGEND = True
	SHOW_BASELINE_MEAN          = False
	SHOW_BASELINE_PER_MODEL     = True
	SLOPE_ARROW_THRESH          = 0.15
	TIER_ORDER                  = ['low','mid-low','mid-high','high']
	BASELINE_LINESTYLE          = (0,(3,2))
	BASELINE_ALPHA              = 0.45
	BASELINE_WIDTH              = 1.5
	LINEWIDTH                   = 1
	MARKERSIZE                  = 6
	SHOW_POINT_TEXT_LABELS      = True   # set True if you want in-plot point labels
	SHOW_COMPLEXITY_LEGEND      = True
	# --- total_cases (shared per tier across models) display config ---
	SHOW_TIER_CASE_BARS      = False
	CASE_BAR_COLOR           = '#888888'
	CASE_BAR_ALPHA           = 0.18
	CASE_BAR_MAX_HEIGHT_PCT  = 22   # reserve vertical span (in sensitivity %) for bars
	SHOW_TIER_CASE_LABELS    = True
	# ------------------------------------------------

	TIER_TO_X = {t:i for i,t in enumerate(TIER_ORDER)}
	X_POSITIONS = np.arange(len(TIER_ORDER))

	def build_complexity_legend_text(qdf):
		"""
		Produce textual explanation for tiers using chosen aggregation mode.
		Tier semantics: low: <Q1, mid-low: Q1–Q2, mid-high: Q2–Q3, high: >Q3.
		"""
		if qdf is None or qdf.empty:
			return "Complexity tiers: low < Q1, mid-low: Q1–Q2, mid-high: Q2–Q3, high > Q3 (quartiles unavailable)."

		q1, q2, q3 = qdf['q1'].mean(), qdf['q2'].mean(), qdf['q3'].mean()

		legend_text = (
			f"Complexity tiers (inference steps $S$; quartiles $Q_1={q1:.0f}, Q_2={q2:.0f}, Q_3={q3:.0f}$): "
			"$\\mathbf{low}$ $(S\\leq Q_1)$; "
			"$\\mathbf{mid\\text{-}low}$ $(Q_1<S\\leq Q_2)$; "
			"$\\mathbf{mid\\text{-}high}$ $(Q_2<S\\leq Q_3)$; "
			"$\\mathbf{high}$ $(S>Q_3)$"
		)


		return legend_text

	def ensure_tier_cat(df):
		df = df.copy()
		if 'tier' not in df.columns:
			raise KeyError("Expected 'tier' column.")
		df['tier'] = (df['tier'].astype(str)
								  .str.strip()
								  .str.lower()
								  .replace({'mid low':'mid-low','mid high':'mid-high'}))
		df.loc[~df['tier'].isin(TIER_ORDER), 'tier'] = pd.NA
		df['tier'] = pd.Categorical(df['tier'], categories=TIER_ORDER, ordered=True)
		return df

	# Normalize models
	long_df['model'] = long_df['model'].map(model_mapping).fillna(long_df['model'])
	overall_df['model'] = overall_df['model'].map(model_mapping).fillna(overall_df['model'])
	long_df = long_df[long_df['model'].isin(ordered_models)]
	overall_df = overall_df[overall_df['model'].isin(ordered_models)]

	model_categories = [m for m in ordered_models if m in long_df['model'].unique()]
	long_df['model'] = pd.Categorical(long_df['model'], categories=model_categories, ordered=True)
	overall_df['model'] = pd.Categorical(overall_df['model'], categories=model_categories, ordered=True)

	long_df = ensure_tier_cat(long_df)

	# Bias ordering
	if getattr(args,'sort_biases','alpha') == 'alpha':
		biases = sorted(long_df['bias'].unique())
	elif args.sort_biases == 'sensitivity':
		slope_rows = []
		for bias, bdf in long_df.groupby('bias'):
			for model, mdf in bdf.groupby('model'):
				mdf = ensure_tier_cat(mdf).sort_values('tier')
				x = mdf['tier'].cat.codes.to_numpy(float)
				y = mdf['tier_sensitivity'].to_numpy(float)
				mask = ~np.isnan(y)
				slope = np.polyfit(x[mask], y[mask], 1)[0] if mask.sum() >= 2 else np.nan
				slope_rows.append({'bias': bias, 'slope': slope})
		bias_mean = (pd.DataFrame(slope_rows)
					 .groupby('bias')['slope']
					 .mean()
					 .sort_values(ascending=False))
		biases = list(bias_mean.index)
	else:
		seen = []
		for b in long_df['bias']:
			if b not in seen: seen.append(b)
		biases = seen

	# Aggregate shared total_cases per (bias, tier)
	tier_counts = (long_df
				   .groupby(['bias','tier'])['tier_total_cases']
				   .first()
				   .unstack())  # index: bias, columns: tier
	# Global scaling for bar heights
	global_max_cases = tier_counts.max().max()

	present_models = model_categories
	palette = sns.color_palette("colorblind", n_colors=len(present_models))
	model_color_map = dict(zip(present_models, palette))

	# Baseline
	baseline = (overall_df[['bias','model','sensitivity']]
				.drop_duplicates()
				.rename(columns={'sensitivity':'baseline_sensitivity'}))
	baseline_mean = (baseline.groupby('bias')['baseline_sensitivity']
							  .mean()
							  .to_dict())

	def compute_model_slopes(bdf_full):
		res = {}
		for model, mdf in bdf_full.groupby('model'):
			mdf = ensure_tier_cat(mdf).sort_values('tier')
			x = mdf['tier'].cat.codes.to_numpy(float)
			y = mdf['tier_sensitivity'].to_numpy(float)
			mask = ~np.isnan(y)
			res[model] = np.polyfit(x[mask], y[mask], 1)[0] if mask.sum() >= 2 else np.nan
		return res

	ncols = 4
	nrows = -(-len(biases)//ncols)
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
							 figsize=(3.8*ncols, 3.1*nrows),
							 constrained_layout=True)
	axes = np.array(axes).reshape(-1)

	legend_handles = {}

	for idx, (ax, bias) in enumerate(zip(axes, biases)):
		bdf = long_df[long_df['bias']==bias]

		# Dense grid
		full_rows = []
		for model in present_models:
			mdf = bdf[bdf['model']==model][['bias','model','tier','tier_sensitivity', 'tier_total_cases']]
			mdf = ensure_tier_cat(mdf).set_index('tier').reindex(TIER_ORDER)
			mdf['bias'] = bias; mdf['model'] = model
			full_rows.append(mdf.reset_index().rename(columns={'index':'tier'}))
		bdf_full = pd.concat(full_rows, ignore_index=True)
		# ---------------- Background bars for shared total_cases ----------------
		if SHOW_TIER_CASE_BARS and bias in tier_counts.index:
			tc_row = tier_counts.loc[bias]
			# y-range reserved: from bottom (0) up to CASE_BAR_MAX_HEIGHT_PCT
			bar_ylim = CASE_BAR_MAX_HEIGHT_PCT
			for ti, tier_name in enumerate(TIER_ORDER):
				cases = tc_row.get(tier_name, np.nan)
				if pd.isna(cases) or cases <= 0 or global_max_cases <= 0:
					continue
				height = bar_ylim * (cases / global_max_cases)
				# draw a rectangle spanning a small horizontal band centered at the tier position
				ax.add_patch(plt.Rectangle(
					(ti - 0.38, 0),   # x, y
					0.76,             # width
					height,           # height
					facecolor=CASE_BAR_COLOR,
					edgecolor='none',
					alpha=CASE_BAR_ALPHA,
					zorder=0.5
				))
				if SHOW_TIER_CASE_LABELS and bias in tier_counts.index:
					tc = tier_counts.loc[bias]
					items = [
						f"{tier}:{int(tc.get(tier))}" + (';' if i < len(TIER_ORDER)-1 else '.')
						for i,tier in enumerate(TIER_ORDER)
						if pd.notna(tc.get(tier))
					]
					# group every 2
					grouped = [" ".join(items[i:i+2]) for i in range(0, len(items), 2)]
					counts_str = "#Dilemmas:\n  " + "\n  ".join(grouped)
					ax.text(
						0.01, 0.99, counts_str,
						transform=ax.transAxes,
						ha='left', va='top',
						fontsize=DEFAULT_FONTSIZE, color='#333',
						bbox=dict(boxstyle="round,pad=0", fc='white', ec='none', alpha=0.75),
						# zorder=100  # ensures it's drawn on top of most elements
					)


					
			# # Visually separate bar zone from sensitivity region (optional guideline)
			# ax.axhline(bar_ylim, color='#999999', linestyle=':', linewidth=0.8, alpha=0.6, zorder=0.6)

		bdf_full = ensure_tier_cat(bdf_full)
		bdf_full = bdf_full.merge(baseline[baseline['bias']==bias],
								  on=['bias','model'], how='left')

		# Baselines per model
		if SHOW_BASELINE_PER_MODEL:
			for model in present_models:
				# baseline value
				y_base = bdf_full.loc[bdf_full['model']==model,'baseline_sensitivity'].dropna()
				if not y_base.empty:
					# white halo first
					ax.axhline(y_base.iloc[0],
							   linestyle=BASELINE_LINESTYLE,
							   linewidth=BASELINE_WIDTH+1.2,
							   color='white', alpha=0.35, zorder=0)
					# colored baseline
					ax.axhline(y_base.iloc[0],
							   linestyle=BASELINE_LINESTYLE,
							   linewidth=BASELINE_WIDTH,
							   color=model_color_map[model],
							   alpha=BASELINE_ALPHA, zorder=1)

		if SHOW_BASELINE_MEAN and bias in baseline_mean:
			ax.axhline(baseline_mean[bias],
					   linestyle=(0,(2,2)),
					   linewidth=2.2, color='gray', alpha=0.6, zorder=1)

		# Plot lines
		for model in present_models:
			mdf = bdf_full[bdf_full['model']==model].sort_values('tier')
			y = mdf['tier_sensitivity'].to_numpy()
			if np.isnan(y).all(): continue
			x = mdf['tier'].astype(str).map(TIER_TO_X).to_numpy()
			line, = ax.plot(x, y, marker='o',
							color=model_color_map[model],
							linewidth=LINEWIDTH,
							markersize=MARKERSIZE,
							label=model)
			if model not in legend_handles:
				legend_handles[model] = line

		# Slopes
		slopes = compute_model_slopes(bdf_full)

		# Axis cosmetics
		ax.set_title(bias.capitalize(), fontsize=DEFAULT_FONTSIZE+2)
		if idx % ncols == 0:
			ax.set_ylabel("Sensitivity (%)")
		if idx // ncols == nrows - 1:
			ax.set_xlabel("Complexity Tier")
		# If you want to visually de-emphasize the bar region, you can tint it:
		ax.axhspan(0, CASE_BAR_MAX_HEIGHT_PCT, facecolor='white', alpha=0.3, zorder=0.4)
		ax.set_ylim(0, 100)

		# Salient ticks
		candidate_ticks = {0,25,50,75,100}
		base_vals = bdf_full['baseline_sensitivity'].dropna().unique()
		for v in base_vals:
			candidate_ticks.add(int(round(v/5.0)*5))
		ticks = sorted(t for t in candidate_ticks if 0 <= t <= 100)
		ax.set_yticks(ticks)
		ax.set_yticklabels([f"{t}%" for t in ticks])

		ax.set_xticks(X_POSITIONS)
		ax.set_xticklabels(TIER_ORDER)
		ax.grid(True, axis='y', linestyle=':', alpha=0.35, zorder=0)

		# Optional endpoint labels
		if SHOW_POINT_TEXT_LABELS:
			used_y = defaultdict(list)

			y_min_sep = 5.0          # minimum vertical separation desired
			max_small_shift = y_min_sep/2    # only apply gentle shift when needed distance <= this
			max_iterations = 6       # safety cap for iterative small nudges

			def can_place(xi, yv):
				return all(abs(yv - uy) >= y_min_sep for uy in used_y[xi])

			def gentle_place(xi, yv):
				"""
				Try to place yv at x index xi.
				If it barely overlaps an existing label (needs ≤ max_small_shift adjustment),
				nudge it up or down minimally. Returns adjusted y or None if cannot place gently.
				"""
				if not used_y[xi]:
					return yv

				# Work on a copy; we'll adjust y_try.
				y_try = yv
				for _ in range(max_iterations):
					distances = [(uy, y_try - uy) for uy in used_y[xi]]
					# Find the closest overlapping label (absolute distance < y_min_sep)
					overlaps = [(uy, diff) for uy, diff in distances if abs(diff) < y_min_sep]
					if not overlaps:
						return y_try  # success

					# Pick the largest magnitude overlap (worst offender) or simply the nearest
					uy, diff = min(overlaps, key=lambda t: abs(t[1]))

					needed = y_min_sep - abs(diff)  # additional separation needed
					if needed > max_small_shift:
						return None  # would require too big a move; abort gentle strategy

					# Direction: if diff > 0, candidate is above uy -> push further up; else push down
					direction = 1 if diff > 0 else -1
					y_try += direction * needed*2

				# If we exit loop without success:
				return None

			for model in present_models:
				mdf = bdf_full[bdf_full['model'] == model].sort_values('tier')
				y_vals = mdf['tier_sensitivity'].to_numpy()
				for xi, yv in enumerate(y_vals):
					if np.isnan(yv):
						continue

					y_final = None
					if can_place(xi, yv):
						y_final = yv
					else:
						# attempt gentle nudge
						y_adj = gentle_place(xi, yv)
						if y_adj is not None and can_place(xi, y_adj):
							y_final = y_adj
						# else: (optional) you could implement a broader search here

					if y_final is not None:
						ax.text(
							xi + 0.05, y_final,
							f"{yv:.1f}%",
							fontsize=DEFAULT_FONTSIZE, ha='left', va='center',
							color=model_color_map[model],
							bbox=dict(boxstyle="round,pad=0.15", fc='white', ec='none', alpha=0.7)
						)
						# Insert into sorted list to keep gentle_place efficient
						used_y[xi].append(y_final)
						used_y[xi].sort()

		# ---- Merged per-panel legend (model + slope ONLY) ----
		if SHOW_PER_PANEL_MODEL_LEGEND:
			legend_entries = []
			legend_handles_local = []
			for model in present_models:
				if model not in legend_handles:
					continue

				# Slope lookup
				s = slopes.get(model, np.nan)
				if np.isnan(s):
					arrow = '·'
					slope_str = "n/a"
				else:
					if s > SLOPE_ARROW_THRESH:
						arrow = '↑'
					elif s < -SLOPE_ARROW_THRESH:
						arrow = '↓'
					else:
						arrow = '→'
					slope_str = f"{arrow}{s:+.2f}"

				label = f"{model}  {slope_str}"
				legend_entries.append(label)
				legend_handles_local.append(legend_handles[model])

			if legend_handles_local:
				ax.legend(legend_handles_local,
						  legend_entries,
						  loc='upper right',
						  fontsize=DEFAULT_FONTSIZE-2,
						  title="Model slope" if idx%4 == 0 else '',
						  title_fontsize=DEFAULT_FONTSIZE-2,
						  frameon=True,
						  framealpha=0.9,
						  borderpad=0.3,
						  handlelength=1.2,
						  handletextpad=0.5)

	# Remove unused axes
	for j in range(len(biases), len(axes)):
		fig.delaxes(axes[j])

	# fig.suptitle('Sensitivity Trends Across Complexity Tiers (Baselines & Merged Legends)', fontsize=16)
	if SHOW_COMPLEXITY_LEGEND:
		legend_text = build_complexity_legend_text(quartiles_df)
		# increase bottom margin to make room for two lines of text
		plt.tight_layout(rect=[0, 0.05, 1, 1])  
		# draw the legend text at y=0.03 (3% up from bottom)
		fig.text(
			0.5,      # centered horizontally
			0.04,     # 3% above bottom
			legend_text,
			ha='center',
			va='bottom',
			fontsize=DEFAULT_FONTSIZE+2,   # bump font size
			# wrap=True
		)

	# plt.tight_layout()
	out_path = f"{output_path_prefix}_lines_with_baseline.pdf"
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	if args.show_figures: plt.show()
	plt.close(fig)

def extract_quartiles_table(df):
	"""
	Extract rows: bias, model, q1, q2, q3 from the 'complexity_analysis' nested dict.
	Returns empty DataFrame if none found.
	"""
	rows = []
	for _, r in df.iterrows():
		comp = r.get('complexity_analysis', {}) or {}
		if isinstance(comp, dict):
			# Any tier's dict should have same quartiles list; grab from first valid tier
			quartiles = None
			for tier_dict in comp.values():
				if isinstance(tier_dict, dict) and 'quartiles' in tier_dict:
					q = tier_dict['quartiles']
					if isinstance(q, (list, tuple)) and len(q) == 3:
						quartiles = q
						break
			if quartiles:
				rows.append({
					'bias': r['bias'],
					'model': r['model'],
					'q1': quartiles[0],
					'q2': quartiles[1],
					'q3': quartiles[2],
				})
	return pd.DataFrame(rows)

def aggregate_complexity_across_biases(long_df, overall_df):
	"""
	Return:
	  agg_mean: model × tier mean sensitivities (ignoring bias)
	  agg_counts: dilemmas counts summed across biases (first non-NA per bias-tier-model taken)
	  model_baseline: overall baseline sensitivity per model (from overall_df)
	  bias_level_values: full distribution (for dist plots)
	"""
	# Ensure only valid tiers
	valid_tiers = ['low','mid-low','mid-high','high']
	long_df = long_df[long_df['tier'].isin(valid_tiers)].copy()

	# Mean sensitivity per model & tier
	agg_mean = (long_df
				.groupby(['model','tier'])['tier_sensitivity']
				.mean()
				.reset_index(name='mean_sensitivity'))

	# Distribution values (retain bias for optional distribution plots)
	bias_level_values = long_df[['bias','model','tier','tier_sensitivity']].dropna()

	# Approx counts: sum (or first non-null) of tier_total_cases across biases
	agg_counts = (long_df
				  .groupby(['model','tier'])['tier_total_cases']
				  .sum(min_count=1)
				  .reset_index(name='total_cases'))

	# Model overall baselines (mean across biases of baseline sensitivity)
	model_baseline = (overall_df[['model','bias','sensitivity']]
					  .drop_duplicates()
					  .groupby('model')['sensitivity']
					  .mean()
					  .to_dict())

	return agg_mean, agg_counts, model_baseline, bias_level_values


def plot_complexity_overall(long_df,
							overall_df,
							output_prefix="2_bias_sensitivity_by_complexity_overall"):
	"""
	Create several plots aggregating complexity tiers over all biases:
	1. Line plot of mean sensitivity vs tier per model
	2. Line plot of Δ from model baseline vs tier
	3. Optional distribution plot (violin) per tier & model
	4.* Heatmap (models × tiers)
	"""
	if long_df.empty:
		print("No complexity_analysis data to aggregate.")
		return

	valid_tiers = ['low','mid-low','mid-high','high']
	long_df = long_df[long_df['tier'].isin(valid_tiers)].copy()

	agg_mean, agg_counts, model_baseline, bias_level_values = aggregate_complexity_across_biases(long_df, overall_df)

	# Sort model order according to ordered_models present
	present_models = [m for m in ordered_models if m in agg_mean['model'].unique()]
	palette = sns.color_palette("colorblind", n_colors=len(present_models))
	color_map = dict(zip(present_models, palette))

	# --- 1. Mean Sensitivity per Tier (Line Plot) ---
	plt.figure(figsize=(8,5))
	for model in present_models:
		m_df = agg_mean[agg_mean['model']==model].set_index('tier').reindex(valid_tiers)
		plt.plot(valid_tiers,
				 m_df['mean_sensitivity'],
				 marker='o',
				 linewidth=2,
				 label=model,
				 color=color_map[model])
	plt.ylabel("Mean Sensitivity (%)")
	plt.xlabel("Complexity Tier")
	# plt.title("Sensitivity vs Complexity (Averaged Across Biases)")
	plt.ylim(0, 100)
	plt.grid(axis='y', linestyle='--', alpha=0.4)
	plt.legend(frameon=True, ncol=2)
	out1 = f"{output_prefix}_mean_lines.pdf"
	plt.savefig(out1, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved {out1}")

	# --- 2. Delta from Model Baseline ---
	labeled = []
	plt.figure(figsize=(8,3))
	for model in present_models:
		m_df = agg_mean[agg_mean['model']==model].set_index('tier').reindex(valid_tiers)
		baseline = model_baseline.get(model, np.nan)
		delta = m_df['mean_sensitivity'] - baseline
		plt.plot(valid_tiers, delta, marker='o', linewidth=2,
				 label=f"{model} (base={baseline:.1f}%)",
				 color=color_map[model])
		
		# Add labels conditionally to avoid overlap
		for x, y in zip(valid_tiers, delta):
			# skip if too close in data‐space to any labeled point
			if any(x==x0 and abs(y - y0) < 0.4 for x0, y0 in labeled):
				continue

			# otherwise draw and record
			plt.text(x, y, f"{y:.1f}%", fontsize=8, color=color_map[model],
					ha='center', va='bottom',
					bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
			labeled.append((x, y))


	plt.axhline(0, color='black', linewidth=1, linestyle=':')
	plt.ylabel("Δ Sensitivity vs Model Baseline")
	plt.xlabel("Complexity Tier")
	# plt.title("Change from Model Baseline Across Complexity")
	plt.grid(axis='y', linestyle='--', alpha=0.4)
	plt.legend(frameon=True, fontsize=DEFAULT_FONTSIZE, ncol=2)
	out2 = f"{output_prefix}_delta_lines.pdf"
	plt.savefig(out2, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved {out2}")

	# --- 3. Distribution Plot (Optional) ---
	if not bias_level_values.empty:
		plt.figure(figsize=(8,3))
		# We can choose violin or box; violin gives shape, box + strip for clarity
		sns.violinplot(data=bias_level_values,
					   x='tier', y='tier_sensitivity',
					   hue='model',
					   order=valid_tiers,
					   palette=color_map,
					   cut=0,
					   linewidth=1,
					   scale='width')
		plt.ylim(0, 100)
		plt.ylabel("Sensitivity (%)")
		plt.xlabel("Complexity Tier")
		# plt.title("Sensitivity Distribution Across Biases by Complexity Tier")
		plt.grid(axis='y', linestyle='--', alpha=0.35)
		plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', title='Model')
		out3 = f"{output_prefix}_violin.pdf"
		plt.savefig(out3, dpi=300, bbox_inches='tight')
		plt.close()
		print(f"Saved {out3}")

	# --- 4. Heatmap (Mean Sensitivity) ---
	heat = (agg_mean
			.pivot(index='model', columns='tier', values='mean_sensitivity')
			.reindex(index=present_models, columns=valid_tiers))
	plt.figure(figsize=(4,0.3*len(present_models)+1.5))
	sns.heatmap(heat, annot=True, fmt=".1f", vmin=0, vmax=100,
				cmap="icefire", cbar_kws={'label':'Mean Sensitivity (%)'})
	# plt.title("Mean Sensitivity by Model & Complexity Tier")
	plt.xlabel("Complexity Tier")
	plt.ylabel("")
	out4 = f"{output_prefix}_heatmap.pdf"
	plt.savefig(out4, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved {out4}")

def plot_high_vs_low_significance(long_df, output_path_prefix="2_bias_complexity_significance", alpha=0.05):
	"""
	Performs paired statistical tests comparing 'high' vs 'low' tier sensitivities per bias,
	computes p-values and confidence intervals for the mean difference, and plots results.
	Also prints the CI values and saves them to CSV.
	"""
	if long_df.empty:
		print("No data for significance testing; skipping plot.")
		return

	# Ensure tier labels standardized
	df = long_df.copy()
	df['tier'] = (df['tier'].astype(str)
					.str.strip()
					.str.lower()
					.replace({'mid low':'mid-low','mid high':'mid-high'}))

	# Filter to only low and high tiers
	df = df[df['tier'].isin(['low', 'high'])]
	results = []

	# Group by bias and test high vs low
	for bias, group in df.groupby('bias'):
		pivot = group.pivot_table(
			index='model',
			columns='tier',
			values='tier_sensitivity'
		)
		# Drop incomplete pairs
		pivot = pivot.dropna(subset=['low', 'high'])
		if pivot.shape[0] < 2:
			# Not enough pairs for test
			continue

		low_vals = pivot['low']
		high_vals = pivot['high']

		# Paired t-test
		tstat, pval = stats.ttest_rel(high_vals, low_vals)
		diff = high_vals - low_vals
		mean_diff = diff.mean()
		sem = stats.sem(diff)
		dfree = len(diff) - 1

		# Confidence interval for mean difference
		ci_lower, ci_upper = stats.t.interval(1 - alpha, dfree, loc=mean_diff, scale=sem)

		results.append({
			'bias': bias,
			'mean_diff': mean_diff,
			'ci_lower': ci_lower,
			'ci_upper': ci_upper,
			'p_value': pval
		})

	if not results:
		print("Not enough data for any bias to test; skipping plot.")
		return

	res_df = pd.DataFrame(results)

	# Sort biases alphabetically and capitalize labels
	res_df['bias'] = res_df['bias'].astype(str).str.strip().str.title()
	res_df = res_df.sort_values('bias').reset_index(drop=True)

	# ---- NEW: print confidence intervals (and save them) ----
	ci_level = 100 * (1 - alpha)
	cols = ['bias', 'mean_diff', 'ci_lower', 'ci_upper', 'p_value']
	print(f"\n{ci_level:.0f}% confidence intervals for mean difference (High − Low):")
	print(
		res_df[cols].to_string(
			index=False,
			formatters={
				'mean_diff': lambda x: f"{x:.3f}",
				'ci_lower':  lambda x: f"{x:.3f}",
				'ci_upper':  lambda x: f"{x:.3f}",
				'p_value':   lambda x: f"{x:.3g}",
			}
		)
	)

	# Also save the table to CSV next to the plot
	res_df.to_csv(f"{output_path_prefix}_stats.csv", index=False)

	# Plotting
	fig, ax = plt.subplots(figsize=(6, 4))

	# Error bars for CI
	y = res_df['mean_diff']
	yerr = np.vstack([
		res_df['mean_diff'] - res_df['ci_lower'],
		res_df['ci_upper'] - res_df['mean_diff']
	])
	ax.errorbar(
		x=res_df['bias'],
		y=y,
		yerr=yerr,
		fmt='o',
		capsize=5,
		linewidth=1.5
	)

	# Annotate p-values and CIs just above the upper CI cap
	y_span = (res_df['ci_upper'].max() - res_df['ci_lower'].min())
	pad = 0.03 * y_span if y_span > 0 else 0.01  # small vertical padding

	for idx, row in res_df.iterrows():
		upper_err = row['ci_upper'] - row['mean_diff']
		ax.text(
			row['bias'],
			row['mean_diff'] + upper_err + pad,
			f"p={row['p_value']:.3f}\nCI=[{row['ci_lower']:.0f}, {row['ci_upper']:.0f}]",
			ha='center',
			va='bottom',
			fontsize=8,
			linespacing=0.9
		)


	ax.axhline(0, color='gray', linestyle='--', linewidth=1)
	ax.set_xticklabels(res_df['bias'], rotation=45, ha='right')
	ax.set_ylabel('High – Low Sensitivity (%)')
	plt.tight_layout()

	out_path = f"{output_path_prefix}.pdf"
	plt.savefig(out_path, dpi=300)
	plt.close(fig)


# -----------------------------
# Main Execution
# -----------------------------

def main():
	df = load_all_results(results_dir)
	print(f"Loaded {len(df)} rows from results.")

	args_str = '_'.join(
		f'{k}={v}'
		for k, v in vars(args).items()
		if parser.get_default(k) != v
	)
	if args_str:
		args_str = '_'+args_str

	# Primary plot
	main_plot_path = os.path.join(results_dir, f'2_bias_sensitivity_plots{args_str}.pdf')
	plot_bias_sensitivities(df, output_path=main_plot_path)
	print(f"Saved main sensitivity plot to {main_plot_path}")

	long_df = expand_complexity_tiers(df)
	if long_df.empty:
		print("No complexity tier data found.")
	else:
		quartiles_df = extract_quartiles_table(df)
		# print(quartiles_df)
		plot_complexity_sensitivity(
			long_df, 
			overall_df=df, 
			quartiles_df=quartiles_df,
			output_path_prefix=os.path.join(results_dir, f'2_bias_sensitivity_by_complexity{args_str}')
		)

		plot_complexity_overall(
			long_df=long_df,
			overall_df=df,
			output_prefix=os.path.join(results_dir, f'2_bias_complexity_overall{args_str}')
		)

		plot_high_vs_low_significance(long_df, os.path.join(results_dir, f'2_bias_sensitivity_stats_by_complexity{args_str}'))

if __name__ == "__main__":
	main()
