import pandas as pd
from scipy.stats import wilcoxon

# Load all files into dataframes
base_df = pd.read_csv('./results/_1st_person_shallow_/sensitivity_scores_base.csv')
cued_df = pd.read_csv('./results/_1st_person_cued_/sensitivity_scores_cued.csv')
cot_df = pd.read_csv('./results/_1st_person_shallow_CoT_/sensitivity_scores_CoT_.csv')
isd_df = pd.read_csv('./results/_1st_person_shallow_impersonified_self_debiasing_/sensitivity_scores_impersonified_self_debiasing_.csv')
ip_df = pd.read_csv('./results/_1st_person_shallow_implication_prompting_/sensitivity_scores_implication_prompting_.csv')
warning_df = pd.read_csv('./results/_1st_person_shallow_warning_/sensitivity_scores_warning_.csv')

# Add a column to identify the technique
base_df['Technique'] = 'Base'
cot_df['Technique'] = 'CoT'
isd_df['Technique'] = 'Imperson. SD'
ip_df['Technique'] = 'Implication Prompting'
warning_df['Technique'] = 'Imperative SD'
cued_df['Technique'] = 'Reason-Cue'

# Concatenate all dataframes
combined_df = pd.concat([base_df, cot_df, isd_df, ip_df, warning_df, cued_df], ignore_index=True)

# Adjusting based on dataframe structure (bias column instead of model)
pivot_df = combined_df.melt(id_vars=['Technique', 'bias'], var_name='Model', value_name='Sensitivity')

# Pivoting for Latex table
latex_table = pivot_df.pivot_table(index=['bias'], columns=['Model', 'Technique'], values='Sensitivity')

# Generate compact LaTeX table with rounded values
latex_output = latex_table.round(1).to_latex(multicolumn=True, multirow=True)

print('LaTeX Table')
print(latex_output)

# Calculate average sensitivity for each technique across all biases and models
average_sensitivity = combined_df.groupby('Technique').mean(numeric_only=True).mean(axis=1).round(3)

# Presenting the results clearly
average_sensitivity_df = pd.DataFrame({
    'Technique': average_sensitivity.index,
    'Average Sensitivity': average_sensitivity.values
}).round(2).sort_values(by='Average Sensitivity', ascending=False)

print("\nAverage Sensitivity:")
print(average_sensitivity_df)

# -----------------------------
# Nonparametric tests: Wilcoxon signed‑rank test comparing each alternative technique to Base
# -----------------------------

# Extract the Base scores in long format
base_long = pivot_df[pivot_df['Technique'] == 'Base'][['bias', 'Model', 'Sensitivity']]

results = []
# Loop over each technique other than Base
techniques = pivot_df['Technique'].unique()
for technique in techniques:
    if technique == 'Base':
        continue
    alt_long = pivot_df[pivot_df['Technique'] == technique][['bias', 'Model', 'Sensitivity']]
    # Merge on 'bias' and 'Model' to ensure paired observations
    merged = pd.merge(base_long, alt_long, on=['bias', 'Model'], suffixes=('_base', '_alt'))
    
    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(merged['Sensitivity_base'], merged['Sensitivity_alt'], alternative='greater')
    results.append({
        'Technique': technique,
        'Wilcoxon statistic': stat,
        'p-value': f"{p_value:.3f}",
        'sigificant': p_value < 0.05/(len(techniques)-1),
    })

results_df = pd.DataFrame(results)
print("\nNonparametric Test Results (Wilcoxon signed‑rank test vs Base):")
print(results_df)