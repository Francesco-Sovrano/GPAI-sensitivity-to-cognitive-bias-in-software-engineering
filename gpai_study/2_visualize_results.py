import os
import re
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

import argparse
parser = argparse.ArgumentParser(description="Provide reasoning, and format")
parser.add_argument(
	"--reasoning",
	type=str,
	required=True,
	help="Specify the format (e.g., 'cued', 'shallow')"
)
parser.add_argument(
	"--format",
	type=str,
	default='1st_person',
	help="Specify the format (e.g., '1st_person', '2nd_person')"
)
parser.add_argument(
	"--bias_warning_in_system_instruction",
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
	"--show_figures",
	action="store_true",
	help="Flag to control whether to show figures (default: False)"
)
args = parser.parse_args()

prompt_label = ''
if args.bias_warning_in_system_instruction:
	prompt_label += 'warning_'
if args.chain_of_thought:
	prompt_label += 'CoT_'
if args.impersonified_self_debiasing:
	prompt_label += 'impersonified_self_debiasing_'
if args.implication_prompting:
	prompt_label += 'implication_prompting_'

csv_file_dir = os.path.join('./results/', f"_{args.format}_{args.reasoning}_"+prompt_label)
img_file_dir = os.path.join(csv_file_dir, "figures")
os.makedirs(img_file_dir, exist_ok=True)

correct_decisions = {
	"overconfidence_bias": ["Option B"], # ["Option A","Option B"],
	"hyperbolic_discounting": ["Option B"],
	"confirmation_bias": ["Option B"],
	"hindsight_bias": ["Option A"],
	"availability_bias": ["Option A"], # ["Option A","Option B"],
	"framing_effect": ["Option A"], # ["Option A","Option B"],
	"bandwagon_effect": ["Option B"],
	"anchoring_bias": ["Option A"], # ["Option A","Option B"],
}

model_mapping = {
	"gpt-3.5-turbo-0125": "gpt-3.5-turbo",
	"gpt-4o-mini-2024-07-18": "gpt-4o-mini",
	"gpt-4o-2024-08-06": "gpt-4o",
	"llama3.3-70b": "llama3.3",
	"llama-3.3-70b-versatile": "llama3.3",
	"deepseek-r1-distill-qwen-32b": "deepseek-r1-qwen",
}

ordered_models = ["llama3.2", "llama3.1", "llama3.3", "deepseek-r1-qwen", "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]

def parse_model_from_filename(filename):
	"""
	Extract the model name from the CSV filename.
	Expected format: decisions_model-<model>_temperature-1_top_p-1.csv
	"""
	basename = os.path.basename(filename)
	match = re.search(r"decisions_model-([^_]+)_", basename)
	if match:
		return match.group(1)
	return "unknown_model"

def load_all_results(results_dir):
	"""Load all CSV files in the results directory and add a model column."""
	csv_files = glob.glob(os.path.join(results_dir, "decisions_*.csv"))
	df_list = []
	for csv_file in csv_files:
		df = pd.read_csv(csv_file)
		df["model"] = parse_model_from_filename(csv_file)
		df_list.append(df)
	if df_list:
		return pd.concat(df_list, ignore_index=True)
	else:
		raise FileNotFoundError("No CSV files found in the results directory.")

def compute_sensitivity(df):
	"""
	Compute sensitivity for each (model, bias) group.
	Only include pairs where the non-biased decision is the correct answer,
	based on a provided mapping from bias to (expected task, correct decision).
	Sensitivity is defined as the percentage of persona-task pairs where the decision
	in the biased scenario differs from the non-biased scenario.
	Returns both the summary and the merged detailed DataFrame.
	"""
	
	# Create a unique identifier for each persona and task combination.
	df["pair_id"] = (df["bias"].astype(str) + "_" + df["task"].astype(str))
	# df["pair_id"] = (df["bias"].astype(str) + "_" + df["task"].astype(str) + "_" +
	# 				 df["gender"].astype(str) + "_" + df["specialty"].astype(str) + "_" +
	# 				 df["experience"].astype(str))
	
	# Separate biased and non-biased decisions and keep the task column.
	biased = df[df["biased_task"] == True].copy()
	nonbiased = df[df["biased_task"] == False].copy()
	
	biased = biased.rename(columns={"decision": "decision_biased"})
	nonbiased = nonbiased.rename(columns={"decision": "decision_nonbiased"})
	
	# Merge the two sets on model and the unique pair identifier.
	merged = pd.merge(biased[["model", "pair_id", "bias", "decision_biased"]],
					  nonbiased[["model", "pair_id", "decision_nonbiased"]],
					  on=["model", "pair_id"])
	
	# # Filter to keep only rows where the non-biased decision is correct.
	# def is_correct(row):
	# 	expected_decision = correct_decisions[row["bias"].split('-')[-1].strip()]
	# 	# Both the task and decision must match.
	# 	return row["decision_nonbiased"] == expected_decision

	# merged = merged[merged.apply(is_correct, axis=1)]
	
	# Now compute if the decisions are different.
	merged["different"] = merged["decision_biased"] != merged["decision_nonbiased"]
	
	# Compute sensitivity (percentage of pairs with differing decisions) per model and bias.
	summary = merged.groupby(["model", "bias"]).agg(
		total_pairs=("different", "count"),
		diff_count=("different", "sum")
	).reset_index()
	summary["sensitivity"] = (summary["diff_count"] / summary["total_pairs"]) * 100
	return summary, merged

def plot_decision_agreement_stacked(df, output_file="decision_agreement_stacked.pdf"):
	"""
	Create a stacked bar plot showing decision agreement for each bias+task pair.
	
	For each unique combination of bias and task (from biased scenarios), the bar is split into
	the number of times "option A" is picked versus "option B" is picked.
	"""
	output_path = os.path.join(img_file_dir, output_file)

	# Filter to consider only biased tasks
	biased_df = df[df["biased_task"] == False].copy()

	def get_formatted_bias_task(row):
		bias_label = row["bias"].split('-')[-1].replace('_', ' ').title()
		task_label = ' '.join(row["task"].split('-')[1:])#.title()
		return f"{bias_label}\n({task_label})"

	# Create a new column that combines bias and task for grouping
	biased_df["bias_task"] = biased_df.apply(get_formatted_bias_task, axis=1)
	# biased_df["bias_task"] = biased_df["bias"] + " - " + biased_df["task"] #+ " - " + str(df["biased_task"])

	# Aggregate counts of each decision per bias+task group
	decision_counts = biased_df.groupby("bias_task")["decision"].value_counts().unstack(fill_value=0)
	print('Decision Counts:', decision_counts)
	
	# Sort the bias_task groups alphabetically (or modify as needed)
	decision_counts = decision_counts.sort_index()
	
	# Ensure the decision columns are in a consistent order.
	# Assuming decisions are labeled "option A" and "option B"
	decision_order = ["Option A", "Option B"]
	available_options = [opt for opt in decision_order if opt in decision_counts.columns]
	
	# Plot the stacked bar chart.
	fig, ax = plt.subplots(figsize=(12, 6))
	decision_counts[available_options].plot(kind='bar', stacked=True, ax=ax, edgecolor='black', width=0.8)
	
	ax.set_ylabel("Count", fontsize=14)
	# ax.set_xlabel("Bias and Task", fontsize=14)
	ax.set_xlabel("")
	ax.set_title("Decision Agreement whithout (Harmful) Bias", fontsize=16, fontweight="bold")
	ax.legend(title="Decision", fontsize=14)
	plt.xticks(rotation=45, ha='right', fontsize=14)
	plt.tight_layout()
	plt.savefig(output_path)
	if args.show_figures:
		plt.show()

def plot_sensitivity_stacked_correct_incorrect(merged, output_file="sensitivity_stacked_correct_incorrect.pdf"):
	"""
	Produce subplots by bias, with stacked bars per model:
	  - Bottom portion = portion of 'different' pairs where no-bias was correct
	  - Top portion    = portion of 'different' pairs where no-bias was incorrect
	The sum of the two stacks is the total sensitivity for (model, bias).

	Changes:
	  - The top portion uses a hatched pattern rather than alpha blending.
	  - Both the correct portion and incorrect portion have inside annotations 
		with their percentage. The top of the bar has the total.
	"""
	output_path = os.path.join(img_file_dir, output_file)

	def is_correct_no_bias(row):
		bias_key = row["bias"].split('-')[-1].strip()
		expected = correct_decisions[bias_key]
		return row["decision_nonbiased"] in expected

	def is_option_b(row):
		return row["decision_nonbiased"] == 'Option B'
	
	merged["no_bias_correct"] = merged.apply(is_correct_no_bias, axis=1)
	merged["chose_option_b"] = merged.apply(is_option_b, axis=1)
	
	# 2) For each (model, bias, no_bias_correct), how many 'different' are True?
	grouped = merged.groupby(["model", "bias", "no_bias_correct"])["different"].agg(
		diff_count="sum",  # how many times we differ
		total_pairs="count"
	).reset_index()
	
	# 3) For each (model, bias), total pairs and portion of difference
	total_pairs_df = grouped.groupby(["model", "bias"])["total_pairs"].sum().reset_index()
	total_pairs_df = total_pairs_df.rename(columns={"total_pairs": "all_pairs"})
	merged_grouped = pd.merge(grouped, total_pairs_df, on=["model", "bias"])
	
	# Fraction of total pairs (percentage)
	merged_grouped["diff_pct"] = (merged_grouped["diff_count"] / merged_grouped["all_pairs"]) * 100
	
	# Pivot so we get separate columns for correct vs. incorrect portion
	pivot_df = merged_grouped.pivot(
		index=["model", "bias"], 
		columns="no_bias_correct",
		values="diff_pct"
	).fillna(0)
	
	# Rename columns: True -> diff_correct_pct, False -> diff_incorrect_pct
	if True in pivot_df.columns:
		pivot_df = pivot_df.rename(columns={True: "diff_correct_pct"})
	else:
		pivot_df["diff_correct_pct"] = 0
	if False in pivot_df.columns:
		pivot_df = pivot_df.rename(columns={False: "diff_incorrect_pct"})
	else:
		pivot_df["diff_incorrect_pct"] = 0
	
	pivot_df = pivot_df.reset_index()
	
	# 4) Subplots by bias, stacked bars per model
	pivot_df["model"] = pivot_df["model"].apply(lambda x: model_mapping.get(x, x))
	
	# Pivot again so we have biases as rows, and columns of models for each portion
	pivot_correct = pivot_df.pivot(index="bias", columns="model", values="diff_correct_pct").fillna(0)
	pivot_incorrect = pivot_df.pivot(index="bias", columns="model", values="diff_incorrect_pct").fillna(0)

	# Print the results in a formatted output
	pivot_total = pivot_correct + pivot_incorrect
	print("Mean sensitivity per model:")
	for model, avg in pivot_total.mean(axis=0).items():
		print(f"{model:15s}: {avg:.2f}%")
	print("Median sensitivity per model:")
	for model, avg in pivot_total.median(axis=0).items():
		print(f"{model:15s}: {avg:.2f}%")

	print("Mean sensitivity per bias:")
	for model, avg in pivot_total.mean(axis=1).items():
		print(f"{model:15s}: {avg:.2f}%")
	print("Median sensitivity per bias:")
	for model, avg in pivot_total.median(axis=1).items():
		print(f"{model:15s}: {avg:.2f}%")
	
	# Reorder columns to match desired order
	pivot_correct = pivot_correct[[m for m in ordered_models if m in pivot_correct.columns]]
	pivot_incorrect = pivot_incorrect[[m for m in ordered_models if m in pivot_incorrect.columns]]
	
	biases = sorted(list(pivot_correct.index), key=lambda x: x.split('-')[-1])
	n_bias = len(biases)
	ncols = min(n_bias, 4)
	nrows = int(math.ceil(n_bias / ncols))
	
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows), sharey=True)
	if n_bias == 1:
		axes = np.array([[axes]])
	axes = axes.flatten()
	
	# Compute maximum total (correct + incorrect)
	total_sens_matrix = pivot_correct.to_numpy() + pivot_incorrect.to_numpy()
	overall_max = np.nanmax(total_sens_matrix)
	y_max = math.ceil(overall_max / 10) * 10 if overall_max > 10 else overall_max + 5
	
	# Color palette for each model
	palette = dict(zip(ordered_models, sns.color_palette("colorblind", len(ordered_models)).as_hex()))
	
	for i, bias in enumerate(biases):
		ax = axes[i]
		
		correct_vals = pivot_correct.loc[bias]
		incorrect_vals = pivot_incorrect.loc[bias]
		models = list(correct_vals.index)
		x_positions = np.arange(len(models))
		
		# We'll store bar handles for legend
		bar_handles = []
		
		for idx, model_name in enumerate(models):
			correct_pct = correct_vals[model_name]
			incorrect_pct = incorrect_vals[model_name]
			total_pct = correct_pct + incorrect_pct
			
			bar_color = palette.get(model_name, "#cccccc")
			
			# Bottom portion (correct)
			bottom_bar = ax.bar(
				x_positions[idx],
				correct_pct,
				color=bar_color,
				edgecolor='black',
				width=0.6
			)
			# Top portion (incorrect), with hatch pattern
			top_bar = ax.bar(
				x_positions[idx],
				incorrect_pct,
				bottom=correct_pct,
				color=bar_color,
				edgecolor='black',
				hatch="///",  # <--- add hatching
				width=0.6
			)
			
			# ----- Annotate each portion -----
			# 1) bottom portion annotation (correct portion), if > 0
			if correct_pct > 1:
				ax.annotate(
					f"{correct_pct + (0 if incorrect_pct > 1 else incorrect_pct):.2f}%",
					xy=(x_positions[idx], correct_pct/2 if correct_pct > 5 else 0),
					ha="center",
					va="center",
					fontsize=11,
					color="black",
					bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1')
				)
			# 2) top portion annotation (incorrect portion), if > 0
			if incorrect_pct > 1:
				ax.annotate(
					f"{incorrect_pct + (0 if correct_pct > 1 else correct_pct):.2f}%",
					xy=(x_positions[idx], (correct_pct + incorrect_pct) if correct_pct > 3 else (2*correct_pct + incorrect_pct)),
					ha="center",
					va="center",
					fontsize=11,
					color="black",
					bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.1')
				)
			
			# 3) total annotation on top
			# if (total_pct != incorrect_pct and total_pct != correct_pct) or total_pct == 0:
			if correct_pct <= 1 and incorrect_pct <= 1:
				ax.annotate(
					f"{total_pct:.2f}%",
					xy=(x_positions[idx], total_pct),
					xytext=(0, 5),
					textcoords="offset points",
					ha='center',
					fontsize=11,
					bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2')
				)
			
			# For legend references (only once per subplot, for example)
			if idx == 0 and i == 0:
				bar_handles = [bottom_bar, top_bar]

		# X-axis labeling
		ax.set_xticks(x_positions)
		ax.set_xticklabels(models, fontsize=14, rotation=25, ha='right')
		
		# Title by last part of bias name
		formatted_bias = bias.split('-')[-1].replace('_', ' ').title()
		ax.set_title(formatted_bias, fontsize=16, fontweight='bold')
		
		if i % ncols == 0:
			ax.set_ylabel("Sensitivity (%)", fontsize=14)
		
		ax.set_ylim(0, y_max)
		ax.grid(axis='y', linestyle='--', alpha=0.7)
	
	# Remove any unused subplots
	for j in range(i+1, len(axes)):
		fig.delaxes(axes[j])
	
	# Create an updated legend:
	# - The non-hatched (non-barred) stacks correspond to instances where in the non-biased (or negatively biased)
	#   task Option A was chosen and in the biased (or positively biased) task Option B was chosen.
	# - The hatched (barred) stacks correspond to instances where in the non-biased (or negatively biased)
	#   task Option B was chosen and in the biased (or positively biased) task Option A was chosen.
	# fig.legend(
	# 	handles=bar_handles,
	# 	# labels=["No-bias correct portion", "No-bias incorrect portion"],
	# 	labels=[
	# 		"Non-barred stacks: non-biased/negatively biased task: Option A chosen; biased/positively biased task: Option B chosen",
	# 		"Barred stacks: non-biased/negatively biased task: Option B chosen; biased/positively biased task: Option A chosen"
	# 	],
	# 	loc='upper right',
	# 	fontsize=10
	# )
	
	plt.tight_layout()
	plt.savefig(output_path)
	if args.show_figures:
		plt.show()

df = load_all_results(csv_file_dir)
print(f"Loaded {len(df)} rows from results.")

# Compute sensitivity metrics.
summary, merged = compute_sensitivity(df)
print("Sensitivity summary:")
print(summary)

# Generate the sensitivity bar plot.
# plot_sensitivity_bar(summary)
plot_sensitivity_stacked_correct_incorrect(merged)

# Generate the stacked bar plot for decision agreement.
plot_decision_agreement_stacked(df)
