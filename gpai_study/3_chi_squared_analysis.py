import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

parser = argparse.ArgumentParser(description="Perform Chi-Squared test on LLM decisions for each bias.")
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help="Specify the model name (e.g., 'llama3.1', 'llama3.1:70b', 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', 'o3-mini-2025-01-31', 'gpt-4o-2024-08-06')"
)
parser.add_argument(
	"--reasoning",
	type=str,
	required=True,
	help="Specify the format (e.g., 'cued', 'shallow')"
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
	"--format",
	type=str,
	default='1st_person',
	help="Specify the format (e.g., '1st_person', '2nd_person')"
)
args = parser.parse_args()

llm_options = {
	'model': args.model,
	'temperature': 1,
	'top_p': 1,
}

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
chi_file_dir = os.path.join(csv_file_dir, "chi_square_results")
os.makedirs(chi_file_dir, exist_ok=True)

# -------------------------------------------------------
# 1. Read the CSV you just saved (adjust the path as needed)
# -------------------------------------------------------
csv_filename = os.path.join(csv_file_dir, f'decisions_{"_".join(map(lambda x: f"{x[0]}-{x[1]}", llm_options.items()))}.csv')

# Read the CSV file with decision data
if not os.path.exists(csv_filename):
	raise FileNotFoundError(f"CSV file {csv_filename} does not exist.")

df = pd.read_csv(csv_filename)

# Ensure output directory exists


# Check that required columns exist
required_cols = {"bias", "biased_task", "decision"}
if not required_cols.issubset(df.columns):
	raise ValueError(f"DataFrame must contain columns {required_cols}, but found {df.columns}.")

# For convenience, ensure "biased_task" is boolean
# (Sometimes CSV can load booleans as strings; just in case.)
if df["biased_task"].dtype != bool:
	# Attempt to interpret 'True'/'False' or '1'/'0' as booleans
	df["biased_task"] = df["biased_task"].astype(str).str.lower().isin(["true", "1"])

# Prepare to store significance results
all_bias_results = []

# Loop over each unique bias in the data
for bias_value in sorted(df["bias"].unique()):
	sub_df = df[df["bias"] == bias_value]

	# Count how many times the model chose Option A vs Option B under
	# Biased vs Non-Biased conditions.
	c11 = sub_df[(sub_df["biased_task"] == True)  & (sub_df["decision"] == "Option A")].shape[0]
	c12 = sub_df[(sub_df["biased_task"] == True)  & (sub_df["decision"] == "Option B")].shape[0]
	c21 = sub_df[(sub_df["biased_task"] == False) & (sub_df["decision"] == "Option A")].shape[0]
	c22 = sub_df[(sub_df["biased_task"] == False) & (sub_df["decision"] == "Option B")].shape[0]

	# Construct the 2x2 contingency table:
	# Rows: [ Biased, Non-Biased ], Columns: [ Option A, Option B ]
	obs = np.array([[c11, c12],
				   [c21, c22]], dtype=float)

	if c11 == c21 and c12 == c22:
		significance = "not significant"
		p_value = 1
		chi2 = 0
		dof = 1
		phi = 0
	else:
		# Perform Chi-Square Test
		chi2, p_value, dof, expected = chi2_contingency(obs)

		# Decide if significant at alpha=0.05
		significance = "SIGNIFICANT" if p_value < 0.05 else "not significant"

		# Compute standardized residuals (2x2)
		# R_ij = (Observed_ij - Expected_ij) / sqrt(Expected_ij)
		with np.errstate(divide='ignore', invalid='ignore'):
			std_res = (obs - expected) / np.sqrt(expected)

		# Compute phi = sqrt(chi^2 / N)
		N = obs.sum()
		phi = (chi2 / N) ** 0.5 if N else 0.0

	# Store textual result
	table_str = f"[[{int(c11)}, {int(c12)}], [{int(c21)}, {int(c22)}]]"
	result_str = (
		f"Bias: {bias_value}\n"
		f"Obs counts (Biased vs Non-Biased, Option A vs Option B): {table_str}\n"
		f"Chi2 = {chi2:.4f}, p-value = {p_value:.4f}, dof = {dof}, => {significance}\n"
		f"phi = {phi:.4f}\n"
		f"{'-'*60}"
	)
	print(result_str)
	all_bias_results.append(result_str)

	# # ---- CREATE A "TABLE" (one figure per bias) ----
	# # Each cell will display:
	# #   Observed
	# #   Expected
	# #   StdResid
	# # For example, "10\n(9.50)\n-0.18"
	# table_data = []
	# for i in range(obs.shape[0]):
	# 	row_data = []
	# 	for j in range(obs.shape[1]):
	# 		row_data.append(
	# 			f"{int(obs[i, j])}\n({expected[i, j]:.2f})\n{std_res[i, j]:.2f}"
	# 		)
	# 	table_data.append(row_data)

	# col_labels = ["Option A", "Option B"]
	# row_labels = ["Biased", "Non-Biased"]

	# fig, ax = plt.subplots()   # Distinct figure, no subplots
	# ax.set_axis_off()          # Hide the axis

	# # Create the table in the center
	# the_table = ax.table(
	# 	cellText=table_data,
	# 	rowLabels=row_labels,
	# 	colLabels=col_labels,
	# 	loc="center"
	# )
	# the_table.scale(1, 2)  # Optionally expand the table a bit

	# # Title shows bias, p-value, effect size
	# plt.title(f"Bias: {bias_value}\nphi={phi:.2f}, p={p_value:.4f} ({significance})")

	# # Save the figure
	# output_file = os.path.join(args.output_dir, f"table_{bias_value}_{args.model}.png")
	# plt.savefig(output_file, bbox_inches="tight")
	# plt.close()

# Optionally, store or log the textual results in a file
results_txt_path = os.path.join(chi_file_dir, f"chi_square_summary_{args.model}.txt")
with open(results_txt_path, "w", encoding="utf-8") as f:
	f.write("\n".join(all_bias_results))
