import os
import pandas as pd
import argparse
import re
import math
import json
from collections import defaultdict, Counter
from more_itertools import unique_everseen

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from lib import *

import random

# set a fixed seed so sampling is reproducible
SEED = 42
random.seed(SEED)

parser = argparse.ArgumentParser(description="Run sensitivity analysis over dilemma prompts using various LLMs")
parser.add_argument(
	"--seed_corpus_only",
	action="store_true",
	help="Restrict evaluation to only the human-seeded dilemmas (filtering out AI-generated ones)"
)
parser.add_argument(
	"--model",
	type=str,
	required=True,
	help=(
		"The LLM to use. "
		"E.g. 'llama3.1', 'llama3.2', 'llama3.3', "
		"'gpt-4o-mini-2024-07-18', 'gpt-3.5-turbo-0125', "
		"'o3-mini-2025-01-31', 'gpt-4o-2024-08-06'"
	)
)
parser.add_argument(
	"--complexity_metric",
	type=str,
	default='inference_steps',
	help="inference_steps or choice_steps"
)
parser.add_argument(
	"--data_model_list",
	nargs="+",
	type=str,
	default=None,
	help=(
		"One or more data model variants to use. "
		"Provide space-separated model names (e.g. 'llama3.1 llama3.2'). "
		"If omitted, defaults to the same single value as --model."
	)
)
parser.add_argument(
	"--n_independent_runs_per_task",
	type=int,
	default=5,
	help="Number of independent runs per dilemma. Default is 5"
)

parser.add_argument(
	"--min_intra_model_agreement_rate_on_dilemma",
	type=float,
	default=0.8,
	help="Minimum intra-model agreement rate required for a dilemma to be considered valid (default: 0.8)"
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
	"--prolog_driven_bistep_axioms_elicitation",
	action="store_true",
)
parser.add_argument(
	"--self_axioms_elicitation",
	action="store_true",
)
#######################################

args = parser.parse_args()
print('Args:', args)

n_independent_runs_per_task = args.n_independent_runs_per_task
complexity_metric = args.complexity_metric

## Normalize data_model_list: if not provided, use the single --model
if args.data_model_list is None:
	data_model_list = [args.model]
else:
	data_model_list = args.data_model_list
model = args.model

if model.startswith('gpt') or model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
	api_key = os.getenv('OPENAI_API_KEY')
	base_url = "https://api.openai.com/v1"
	parallelise = True
elif model in ['deepseek-r1-distill-llama-70b','llama-3.3-70b-versatile','llama-3.1-8b-instant', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b']:
	api_key = os.getenv('GROQ_API_KEY')
	base_url = "https://api.groq.com/openai/v1"
	parallelise = True
else:
	api_key ='ollama' # required, but unused
	base_url = 'http://localhost:11434/v1'
	parallelise = False

llm_options = {
	'model': model,
	'temperature': args.temperature,
	'top_p': args.top_p,
	'api_key': api_key,
	'base_url': base_url,
	'parallelise': parallelise,
}

_auto_cue_instruction = """I have a dilemma described in natural language (NL), and I want you to shortly describe what software engineering (SE) best practices are related to the dilemma and how, without mentioning any of the options.

When responding, your answer must be formatted exactly as follows:
Best Practices: <a short description of the best practices>

Make sure to follow this format strictly without any additional text or formatting."""


# _auto_cue_instruction = """I have a dilemma described in natural language (NL), and I want you to shortly describe what software engineering (SE) best practices are related to the dilemma and how, without mentioning any of the options. The best practices must be generic: neither bound to the specifics of the dilemma nor tied to either of the two options.

# When responding, your answer must be formatted exactly as follows:
# Best Practices: <a short description of the best practices>

# Make sure to follow this format strictly without any additional text or formatting."""

_prolog_based_auto_cue_instruction = """I have a logical dilemma described in natural language (NL), and I want you to translate it into a compact Prolog program so that my Prolog reasoner can solve the dilemma. The reasoner should be able to understand on its own what the appropriate decision is. Hence, please define suitable axioms based on software engineering (SE) best practices to allow the reasoner to infer the optimal decision independently. The final decision must be either "option_A" or "option_B", and the instruction to run at the end must be "decide_option(user, Choice)." which shall return the final decision.
The Prolog program must also:
- include all the facts mentioned in the NL description;
- not explain or provide hints whatsoever about what's the correct option;
- not include any comments.

Start by writing the axioms, which must be generic: neither bound to the specifics of the dilemma nor tied to either of the two options. Once the axioms are complete, proceed to write the rest of the program, including facts and dilemma-specific statements. Finally, provide a short natural language representation of the axioms, avoiding sections or subsections, and ending with a sentence that begins with: "The best practice is:" without mentioning any option.

Examples of NL Representations of Axioms are:
1. `Thorough stakeholder interviews ensure that requirements are complete, while a lack of thorough interviews creates a risk of costly rework. Tight launch deadlines only amplify that risk, so whenever high rework risk arises under a tight schedule, you should refine and detail your requirements through additional interviews. The best practice is: ensure requirements are complete by conducting thorough stakeholder interviews early and, if deadlines tighten and rework risk grows, immediately refine requirements with detailed follow-up interviews.`
2. `A modular design makes the system easier to maintain. Introducing a caching layer requires validation through performance tests. Optimizing SQL queries must be confirmed with regression tests. Comprehensive testing includes unit tests for both common cases and edge cases, integration tests, performance tests, and regression tests. The best practice is: maintain a modular design, automate the CI/CD pipeline, and validate every change with a full suite of automated tests.`

Your output must adhere to this template:
## Axioms
```prolog
% code here
```

## Facts
```prolog
:- consult('axioms').
% rest of the code here
```

## NL Representation of Axioms
`...`
"""

_system_instruction = (
	"When responding to any decision-making task, your answer must be formatted exactly as follows:\n"
	# "Explanation: <a very short explanation of your decision here>\n"
	"Explanation: <a short explanation of your decision here>\n"
	"Decision: <your decision here: either Option A or Option B>\n"
	"Make sure to follow this format strictly without any additional text or formatting."
)

if args.self_axioms_elicitation:
	_system_instruction += "The Explanation must first shortly describe what software engineering (SE) best practices are related to the dilemma and how. Then it must use them to justify the decision accordingly."

if args.bias_warning:
	_system_instruction += "\nMake sure your reasoning is not influenced by any cognitive bias."
	_auto_cue_instruction += "\nMake sure your reasoning is not influenced by any cognitive bias."
	_prolog_based_auto_cue_instruction += "\nMake sure your reasoning is not influenced by any cognitive bias."

if args.impersonified_self_debiasing:
	_system_instruction = "You are an unbiased software engineer that is not affected by biased statements.\n" + _system_instruction
	_auto_cue_instruction = "You are an unbiased software engineer that is not affected by biased statements.\n" + _auto_cue_instruction
	_prolog_based_auto_cue_instruction = "You are an unbiased software engineer that is not affected by biased statements.\n" + _prolog_based_auto_cue_instruction

if args.implication_prompting:
	_system_instruction += "\nThinking step by step, provide the implication of the given decision, explaining why your decision might be biased."
	_auto_cue_instruction += "\nThinking step by step, provide the implication of the given decision, explaining why your decision might be biased."
	_prolog_based_auto_cue_instruction += "\nThinking step by step, provide the implication of the given decision, explaining why your decision might be biased."

if args.chain_of_thought:
	_system_instruction += "\nBreak the reasoning into steps, and output the result of each step as you perform it."
	_auto_cue_instruction += "\nBreak the reasoning into steps, and output the result of each step as you perform it."
	_prolog_based_auto_cue_instruction += "\nBreak the reasoning into steps, and output the result of each step as you perform it."


def compute_average_levenshtein_similarity_of_text_list(texts):
	num_texts = len(texts)
	# Initialize similarity matrix
	similarity_matrix = np.zeros((num_texts, num_texts), dtype=np.float64)

	# Compute normalized Levenshtein similarity for each pair
	for i in range(num_texts):
		for j in range(num_texts):
			if i != j:
				dist = Levenshtein.distance(texts[i], texts[j])
				max_len = max(len(texts[i]), len(texts[j]))
				# Avoid division by zero
				if max_len == 0:
					similarity = 1.0
				else:
					similarity = 1 - dist / max_len  # Normalize to similarity score
				similarity_matrix[i][j] = similarity
			else:
				similarity_matrix[i][j] = np.nan  # Mask self-comparisons

	# Compute per-text average similarity (excluding self)
	per_text_avg = np.nanmean(similarity_matrix, axis=1)
	# Compute overall average similarity
	overall_avg = np.nanmean(per_text_avg)
	return float(overall_avg)

def compute_average_semantic_similarity_of_text_list(texts):
	# Compute embeddings
	embeddings = SEMANTIC_SIMILARITY_MODEL.encode(texts)  # Returns a NumPy array
	# Compute pairwise cosine similarity matrix
	cosine_sim_matrix = cosine_similarity(embeddings)
	# Mask diagonal (self-similarity) to exclude it from average
	np.fill_diagonal(cosine_sim_matrix, np.nan)
	# Compute per-text average similarity (excluding self)
	per_text_avg = np.nanmean(cosine_sim_matrix, axis=1)
	# Compute overall average similarity
	overall_avg = np.nanmean(per_text_avg)
	return float(overall_avg)

def get_rules_tier(count):
	if count <= q1:
		return 'low'
	elif count <= q2:
		return 'mid-low'
	elif count <= q3:
		return 'mid-high'
	else:
		return 'high'

def get_best_practices_from_output(gpai_output):
	if not gpai_output:
		return ''
	decision_pattern = r'[*#\s"\'()\n]*Best Practices[*#\s"\'()\n]*[:\n][*#\s"\'\n]*([^\n]+)[*#\s"\']*'
	
	# Find all matches for the decision pattern
	cs_matches = list(re.finditer(decision_pattern, gpai_output, re.IGNORECASE))
	# Extract the last match if it exists
	decision = cs_matches[-1].group(1).strip().strip('.*') if cs_matches else None
	if not decision:
		print(f'Cannot get a decision for: {gpai_output}')
		return ''
	# print(decision)
	return decision

def extract_axioms_description(text):
	if not text:
		return ''
	pattern = re.compile(
		r"^.*?Axioms[^\n]*\n+[`]+prolog(?P<axioms>[^`]+)[`]+.*?"
		r"^.*?Facts[^\n]*\n+[`]+prolog(?P<facts>[^`]+)[`]+.*?"
		r"^.*?NL[^\n]*\n+[`]*(?P<axioms_description>[^\n`]+)[`]*",
		re.DOTALL | re.MULTILINE
	)
	for m in re.finditer(pattern, text):
		axioms, facts, axioms_description = m.group('axioms').strip(), m.group('facts').strip(), m.group('axioms_description').strip()
		if not axioms or not facts or not axioms_description:
			print(f'Cannot get a decision for: {text}')
			print('-'*10)
			print(axioms)
			print('-'*10)
			print(facts)
			print('-'*10)
			print(axioms_description)
			return ''
		# if not 'best practice is:' in axioms_description:
		# 	print(f'Cannot find best practice:', axioms_description)
		# 	return axioms_description
		# print(axioms_description)
		return axioms_description#.split('best practice is:')[-1]
	print(f'Cannot find patterns for: {text}')
	return ''

dilemmas_dataset = defaultdict(list)
for data_model in data_model_list:
	with open(f"./dataset/augmented_dilemmas_dataset_{data_model}.json", "r", encoding="utf-8") as f:
		this_dilemmas_dataset = json.load(f)
	for k,v in this_dilemmas_dataset.items():
		dilemmas_dataset[k] += v
# Remove duplicated dilemmas across separated datasets
if len(data_model_list) > 1:
	for k in dilemmas_dataset.keys():
		vs = dilemmas_dataset[k]
		index_set_of_duplicated_texts,_ = get_index_set_of_duplicated_texts(list(map(lambda x: x["unbiased"], vs)))
		dilemmas_dataset[k] = [
			v
			for i,v in enumerate(vs)
			if i not in index_set_of_duplicated_texts
		]

choice_steps_list = [d[complexity_metric] for dilemma_list in dilemmas_dataset.values() for d in dilemma_list]
q1, q2, q3 = map(float,np.percentile(choice_steps_list, [25, 50, 75]))

min_dilemma_list = min(map(len, dilemmas_dataset.values()))
sensitivity_dict = {}
results = []
for bias_name, dilemma_list in dilemmas_dataset.items():
	this_results = []
	if args.seed_corpus_only:
		capped_dilemma_list = list(filter(lambda x: not x["AI_generated"], dilemma_list))
	else:
		seed_corpus = list(filter(lambda x: not x["AI_generated"], dilemma_list))
		ai_corpus = list(filter(lambda x: x["AI_generated"], dilemma_list))
		capped_dilemma_list = random.sample(ai_corpus, min_dilemma_list-len(seed_corpus))+seed_corpus # enforce the same number of testing dilemmas across all biases for a fair comparison

	if args.bistep_axioms_elicitation or args.prolog_driven_bistep_axioms_elicitation:
		_map_fn = get_best_practices_from_output if args.bistep_axioms_elicitation else extract_axioms_description
		_instruction = _auto_cue_instruction if args.bistep_axioms_elicitation else _prolog_based_auto_cue_instruction
		biased_output_list = instruct_model([
			dilemma['biased']
			for dilemma in capped_dilemma_list
		], system_instructions=[_instruction]*len(capped_dilemma_list), **llm_options)
		biased_task_list = [
			dilemma['biased']+'\n\nReasoning cues:\n'+axioms_description+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma, axioms_description in zip(capped_dilemma_list, map(_map_fn, biased_output_list))
		]
		
		unbiased_output_list = instruct_model([
			dilemma['unbiased']
			for dilemma in capped_dilemma_list
		], system_instructions=[_instruction]*len(capped_dilemma_list), **llm_options)
		unbiased_task_list = [
			dilemma['unbiased']+'\n\nReasoning cues:\n'+axioms_description+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma, axioms_description in zip(capped_dilemma_list, map(_map_fn, unbiased_output_list))
		]
	elif args.inject_axioms:
		biased_task_list = [
			dilemma['biased']+'\n\nReasoning cues:\n'+dilemma['axioms_description'].split('best practice is:')[-1]+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
		unbiased_task_list = [
			dilemma['unbiased']+'\n\nReasoning cues:\n'+dilemma['axioms_description'].split('best practice is:')[-1]+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
	elif args.inject_axioms_in_prolog:
		biased_task_list = [
			dilemma['biased']+'\n\nProlog-encoded reasoning cues:\n'+dilemma['axioms']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
		unbiased_task_list = [
			dilemma['unbiased']+'\n\nReasoning cues:\n'+dilemma['axioms']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
	else:
		biased_task_list = [
			dilemma['biased']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]
		unbiased_task_list = [
			dilemma['unbiased']+' '*i
			for i in range(n_independent_runs_per_task)
			for dilemma in capped_dilemma_list
		]

	expanded_dilemma_list = [
		dilemma
		for i in range(n_independent_runs_per_task)
		for dilemma in capped_dilemma_list
	]

	# print('Mean semantic similarity of unbiased tasks:', compute_average_semantic_similarity_of_text_list(unbiased_task_list))

	biased_output_list = instruct_model(biased_task_list, system_instructions=[_system_instruction]*len(biased_task_list), **llm_options)
	unbiased_output_list = instruct_model(unbiased_task_list, system_instructions=[_system_instruction]*len(unbiased_task_list), **llm_options)

	unbiased_dilemma_to_decision_dict = defaultdict(list)
	for biased_output, biased_prompt, unbiased_output, unbiased_prompt, dilemma in zip(biased_output_list, biased_task_list, unbiased_output_list, unbiased_task_list, expanded_dilemma_list):
		# if args.inject_axioms:
		# 	if 'best practice is:' not in dilemma['axioms_description']:
		# 		continue
		try:
			biased_decision, biased_decision_explanation = get_decision_and_explanation_from_output(biased_output, bias_name)
			unbiased_decision, unbiased_decision_explanation = get_decision_and_explanation_from_output(unbiased_output, bias_name)
		except Exception as e:
			print(f"<{bias_name}> Error: {e}")
			print('-'*10)
			print('biased_output:', biased_output)
			print('-'*10)
			print('unbiased_output:', unbiased_output)
			print('#'*10)
			continue

		bias_was_harmful = False
		sensitive_to_bias = False
		unbiased_decision_differs_from_expected_decision = unbiased_decision[-1] != dilemma['correct_option'][-1]
		if unbiased_decision_differs_from_expected_decision:
			if unbiased_decision == biased_decision:
				bias_was_harmful = True
		else:	
			if unbiased_decision != biased_decision:
				bias_was_harmful = True
		if unbiased_decision != biased_decision:
			sensitive_to_bias = True

		this_results.append({
			'bias_name': bias_name,
			'bias_was_harmful': bias_was_harmful,
			'sensitive_to_bias': sensitive_to_bias,
			'unbiased_decision_differs_from_expected_decision': unbiased_decision_differs_from_expected_decision,
			'suggested_decision_with_bias': biased_decision,
			'suggested_decision_without_bias': unbiased_decision,
			'decision_explanation_with_bias': biased_decision_explanation,
			'decision_explanation_without_bias': unbiased_decision_explanation,
			'prompt_with_bias': biased_prompt,
			'prompt_without_bias': unbiased_prompt,
			**dilemma
		})

	harmful_decisions = 0
	differing_decisions = 0
	total_cases = 0
	prolog_vs_model_disagreement_on_unbiased = 0
	for r in this_results:
		# if r['bias_name'] != bias_name:
		# 	continue
		if r['unbiased_decision_differs_from_expected_decision']:
			unbiased_decision_differs_from_expected_decision += 1
		if r['sensitive_to_bias']:
			differing_decisions += 1
		if r['bias_was_harmful']:
			harmful_decisions += 1
		total_cases += 1

	# Calculate sensitivity for this bias (percentage of differing decisions)
	# total_cases = len(personas) * len(task_dirs)
	harmfulness = (harmful_decisions / total_cases)*100
	sensitivity = (differing_decisions / total_cases) * 100
	prolog_uncertainty = (unbiased_decision_differs_from_expected_decision / total_cases) * 100
	# print(f"Bias: {bias_name}, Sensitivity: {sensitivity:.2f}% based on {total_cases} cases.")
	sensitivity_dict[bias_name] = {
		'sensitivity': sensitivity,
		'harmfulness': harmfulness,
		'total_runs': total_cases,
		'prolog_uncertainty': prolog_uncertainty,
		'average_semantic_similarity_of_dilemmas': compute_average_semantic_similarity_of_text_list([
			dilemma['unbiased']
			for dilemma in capped_dilemma_list
		]),
		'average_levenshtein_distance_of_dilemmas': compute_average_levenshtein_similarity_of_text_list([
			dilemma['unbiased']
			for dilemma in capped_dilemma_list
		]),
	}

	tiered_results = defaultdict(list)
	for r in this_results:
		# if r['bias_name'] != bias_name:
		# 	continue
		tier = get_rules_tier(r[complexity_metric])
		tiered_results[tier].append(r)

	sensitivity_by_rules_tier = {}
	for tier, tier_results in tiered_results.items():
		total = len(tier_results)
		if total == 0:
			continue
		harmful = sum(r['bias_was_harmful'] for r in tier_results)
		sensitive = sum(r['sensitive_to_bias'] for r in tier_results)
		uncertain = sum(r['unbiased_decision_differs_from_expected_decision'] for r in tier_results)

		sensitivity_by_rules_tier[tier] = {
			'total_runs': total,
			'sensitivity': (sensitive / total) * 100,
			'harmfulness': (harmful / total) * 100,
			'prolog_uncertainty': (uncertain / total) * 100,
			'quartiles': [q1,q2,q3],
		}

	sensitivity_dict[bias_name]['complexity_analysis'] = sensitivity_by_rules_tier
	results += this_results

args_str = '.'.join(
	f'{k}={v}'
	for k, v in vars(args).items()
	if parser.get_default(k) != v
)
os.makedirs("./generated_output_data", exist_ok=True)

df = pd.DataFrame(results)
df.to_csv(f'./generated_output_data/1_llm_outputs_{args_str}.csv', index=False)

# with open(f"./generated_output_data/llm_outputs_{model}.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=4, ensure_ascii=False)

with open(f"./generated_output_data/1_bias_sensitivity_{args_str}.json", "w", encoding="utf-8") as f:
	json.dump(sensitivity_dict, f, indent=4, ensure_ascii=False)

print(json.dumps(sensitivity_dict, indent=4))