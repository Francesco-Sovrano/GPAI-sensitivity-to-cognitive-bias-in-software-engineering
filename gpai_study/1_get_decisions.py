import os
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict
import json
import pandas as pd
import re
from lib import *

import argparse
parser = argparse.ArgumentParser(description="Provide model, reasoning, format, and how many independent runs to have per task")
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help="Specify the model name (e.g., 'llama3.1', 'llama3.2', 'llama3.3', 'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', 'o3-mini-2025-01-31', 'gpt-4o-2024-08-06')"
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
	help="Specify the format: '1st_person', '2nd_person'"
)
parser.add_argument(
	"--number_of_independent_runs_per_task",
	type=int,
	default=100,
	help="Specify how many independent runs to have per task. Default is 100"
)
args = parser.parse_args()

scenarios_dir = os.path.join('../scenarios', f"_{args.format}_{args.reasoning}_")

number_of_independent_runs_per_task = args.number_of_independent_runs_per_task

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
os.makedirs(csv_file_dir, exist_ok=True)

model = args.model
if model.startswith('gpt') or model.startswith('o1') or model.startswith('o3'):
	api_key = os.getenv('OPENAI_API_KEY')
	base_url = "https://api.openai.com/v1"
	parallelise = True
elif model in ['deepseek-r1-distill-qwen-32b','llama-3.3-70b-versatile','mixtral-8x7b-32768']:
	api_key = os.getenv('GROQ_API_KEY')
	base_url = "https://api.groq.com/openai/v1"
	parallelise = True
else:
	api_key ='ollama' # required, but unused
	base_url = 'http://localhost:11434/v1'
	parallelise = False

llm_options = {
	'model': model,
	'temperature': 1,
	'top_p': 1,
	'api_key': api_key,
	'base_url': base_url,
	'parallelise': parallelise,
}

system_instruction = (
	"When responding to any decision-making task, your answer must be formatted exactly as follows:\n"
	# "Explanation: <a very short explanation of your decision here>\n"
	"Explanation: <a cued explanation of your decision here>\n"
	"Decision: <your decision here: either Option A or Option B>\n"
	"Make sure to follow this format strictly without any additional text or formatting."
)
if args.bias_warning_in_system_instruction:
	system_instruction += "\nMake sure your reasoning is not influenced by any cognitive bias."

if args.impersonified_self_debiasing:
	system_instruction = "You are an unbiased software engineer that is not affected by biased statements.\n" + system_instruction

if args.implication_prompting:
	system_instruction += "\nThinking step by step, provide the implication of the given decision, explaining why your decision might be biased."

if args.chain_of_thought:
	system_instruction += "\nBreak the reasoning into steps, and output the result of each step as you perform it."

# ----------------------------
# 2. Process the scenarios
# ----------------------------

decision_pattern = r'[*#\s"\'()\n]*Decision[*#\s"\'()\n]*[:\n][*#\s"\'\n]*([^\n]+)[*#\s"\']*'
explanation_pattern = r'[*#\s"\'()\n]*Explanation[*#\s"\'()\n]*[:\n][*#\s"\'\n]*([^\n]+)[*#\s"\']*'

def get_decision_and_explanation_from_output(gpai_output, bias, task):
	# Find all matches for the decision pattern
	cs_matches = list(re.finditer(decision_pattern, gpai_output.replace('Answer','Decision').replace('Option:','Decision:')))
	# Extract the last match if it exists
	decision = cs_matches[-1].group(1).strip().strip('.*') if cs_matches else None
	# Find all matches for the explanation pattern
	se_matches = list(re.finditer(explanation_pattern, gpai_output))
	# Extract the last match if it exists
	explanation = se_matches[-1].group(1).strip().strip('.*') if se_matches else ''
	# # Regular expressions to match values after 'Decision:' and 'Explanation:'
	# cs_match = re.search(decision_pattern, gpai_output)
	# se_match = re.search(explanation_pattern, gpai_output)
	# # Extracting the values if matches are found
	# decision = cs_match.group(1).strip().strip('.') if cs_match else None
	# explanation = se_match.group(1).strip().strip('.') if se_match else ''
	if not decision:
		# print(decision)
		raise ValueError(f'Cannot get a decision for: {gpai_output}')
	if decision.casefold().startswith('Option A'.casefold()) or ('Option A'.casefold() in decision.casefold() and 'Option B'.casefold() not in decision.casefold()) or decision.casefold() == 'A'.casefold():
		decision = 'Option A'
	elif decision.casefold().startswith('Option B'.casefold()) or ('Option B'.casefold() in decision.casefold() and 'Option A'.casefold() not in decision.casefold()) or decision.casefold() == 'B'.casefold():
		decision = 'Option B'
	else:
		if bias == 'memory - hindsight_bias':
			if decision.casefold() == 'Inappropriate'.casefold():
				decision = 'Option B'
			elif decision.casefold() == 'Appropriate'.casefold():
				decision = 'Option A'
			else:
				# print(decision)
				decision = None
		else:
			# print(decision)
			decision = None
	if not decision:
		raise ValueError(f'Cannot get a decision for: {gpai_output}')
	return decision, explanation

# We assume that the top-level extracted folder (scenarios_dir) contains one directory per bias.
bias_dirs = [d for d in os.listdir(scenarios_dir) if os.path.isdir(os.path.join(scenarios_dir, d))]
results = []
for bias in bias_dirs:
	bias_path = os.path.join(scenarios_dir, bias)
	# Assume two decision-making tasks per bias
	task_dirs = [d for d in os.listdir(bias_path) if os.path.isdir(os.path.join(bias_path, d))]
	
	# To record differences: count total cases and count where decisions differ
	total_cases = 0
	differing_decisions = 0
	
	for task in task_dirs:
		task_path = os.path.join(bias_path, task)
		# List txt files: we assume one file is the biased version and one is less/non biased.
		txt_files = [f for f in os.listdir(task_path) if f.endswith('.txt')]
		if len(txt_files) != 2:
			print(f"Unexpected number of txt files in {task_path}")
			continue
		
		if txt_files[0].startswith('0'):
			nonbiased_file = txt_files[0]
			biased_file = txt_files[-1]
		elif txt_files[-1].startswith('0'):
			nonbiased_file = txt_files[-1]
			biased_file = txt_files[0]
		
		if not (biased_file and nonbiased_file):
			print(f"Could not determine biased vs non-biased in {task_path}")
			continue
		
		with open(os.path.join(task_path, biased_file), 'r', encoding='utf-8') as file:
			vignette_biased = file.read()
		with open(os.path.join(task_path, nonbiased_file), 'r', encoding='utf-8') as file:
			vignette_nonbiased = file.read()
		
		system_instruction_list = [
			system_instruction+' '*i
			for i in range(number_of_independent_runs_per_task)
		]
		biased_outputs = instruct_model([vignette_biased]*len(system_instruction_list), system_instructions=system_instruction_list, **llm_options)
		nonbiased_outputs = instruct_model([vignette_nonbiased]*len(system_instruction_list), system_instructions=system_instruction_list, **llm_options)

		for biased, nonbiased in zip(biased_outputs, nonbiased_outputs):
			try:
				biased_decision, biased_decision_explanation = get_decision_and_explanation_from_output(biased, bias, task)
				nonbiased_decision, nonbiased_decision_explanation = get_decision_and_explanation_from_output(nonbiased, bias, task)
			except Exception as e:
				print(bias, task, e)
				print('#'*10)
				continue
			results.append({
				'decision': biased_decision,
				'explanation': biased_decision_explanation,
				'biased_task': True,
				'bias': bias,
				'task': task,
			})
			results.append({
				'decision': nonbiased_decision,
				'explanation': nonbiased_decision_explanation,
				'biased_task': False,
				'bias': bias,
				'task': task,
			})
			if nonbiased_decision != biased_decision:
				differing_decisions += 1
			total_cases += 1
				
	# Calculate sensitivity for this bias (percentage of differing decisions)
	# total_cases = len(personas) * len(task_dirs)
	sensitivity = (differing_decisions / total_cases) * 100
	print(f"Bias: {bias}, Sensitivity: {sensitivity:.2f}% based on {total_cases} cases.")

df = pd.DataFrame(results)
# Define the filename for the CSV file
short_llm_options = {key: llm_options[key] for key in ['model', 'temperature', 'top_p'] if key in llm_options}
csv_filename = os.path.join(csv_file_dir, f'decisions_{"_".join(map(lambda x: f"{x[0]}-{x[1]}", short_llm_options.items()))}.csv')
# Save the DataFrame to a CSV file
df.to_csv(csv_filename, index=False)

# print(json.dumps(results, indent=4))

