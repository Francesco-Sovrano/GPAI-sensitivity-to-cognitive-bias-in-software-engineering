import os
import pandas as pd


model_name = 'gpt-4o-mini' # 'llama-3.1-8b'
extract_dir = 'data_to_analyze'

# File paths
file1 = os.path.join(extract_dir, model_name, f"1_llm_outputs_model={model_name}.data_model_list=['gpt-4.1-mini', 'gpt-4o-mini', 'deepseek-r1-distill-llama-70b'].csv")
file2 = os.path.join(extract_dir, model_name, f"1_llm_outputs_model={model_name}.data_model_list=['gpt-4.1-mini', 'gpt-4o-mini', 'deepseek-r1-distill-llama-70b'].bias_warning=True.self_axioms_elicitation=True.csv")

# Load data
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Step 1: filter df1 where sensitive_to_bias == True
df1_filtered = df1[df1['sensitive_to_bias'] == True]

# Step 2: filter df2 where sensitive_to_bias == False
df2_filtered = df2[df2['sensitive_to_bias'] == False]

# Step 3: find common entries (intersection). Use 'pair' and 'run_id' as likely keys for joining
merged = pd.merge(
    df1_filtered, 
    df2_filtered, 
    on=['bias_name','prompt_with_bias', 'prompt_without_bias'], 
    how='inner',
    suffixes=('_BASE', '_OURS')
)

merged = merged[[
    'bias_name', 
    'prompt_with_bias',
    'prompt_without_bias', 
    'decision_explanation_with_bias_BASE', 
    'decision_explanation_without_bias_BASE', 
    'decision_explanation_with_bias_OURS', 
    'decision_explanation_without_bias_OURS'
]]

# Strip whitespace from both columns
merged['prompt_with_bias'] = merged['prompt_with_bias'].str.strip()
merged['prompt_without_bias'] = merged['prompt_without_bias'].str.strip()
merged = merged.drop_duplicates(subset=['prompt_with_bias', 'prompt_without_bias'])

merged.to_csv(f'data_from_{model_name}_to_analyze.csv', index=False)
