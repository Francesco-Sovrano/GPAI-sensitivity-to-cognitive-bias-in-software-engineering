import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)      # current = (256, 10240) on macOS
new_soft = min(hard, 40960)                                  # never exceed hard
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard)) # requires sudo if > hard

import Levenshtein
import re
import pandas as pd
import json
import pickle
from tqdm import tqdm
import multiprocessing
from more_itertools import unique_everseen
import concurrent.futures
import copy
import ollama
import openai

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import tempfile
import subprocess
import sys

SEMANTIC_SIMILARITY_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

DECISION_GENERATION_SYSTEM_INSTRUCTION = (
	"When responding to any decision-making task, your answer must be formatted exactly as follows:\n"
	# "Explanation: <a very short explanation of your decision here>\n"
	"Explanation: <a short explanation of your decision here>\n"
	"Decision: <your decision here: either Option A or Option B>\n"
	"Make sure to follow this format strictly without any additional text or formatting."
)

MAX_SIMILARITY_THRESHOLD_FOR_DILEMMAS = 0.90

TEXT2PROLOG_PROMPT_TEMPLATE = """## Unbiased NL Dilemma
{unbiased_dilemma_description}

## Biased NL Dilemma
{biased_dilemma_description}"""

TEXT2PROLOG_SYSTEM_INSTRUCTION = """I have two versions of the same logical dilemma, one version exhibits a {bias_label}, while the other does not. Each dilemma is described in natural language (NL) as a reasoning problem, and I want you to translate each one into a compact Prolog program so that my Prolog reasoner can solve the dilemma. The reasoner should be able to understand on its own what the appropriate decision is. Hence, please define suitable axioms based on software engineering (SE) best practices to allow the reasoner to infer the optimal decision independently. The final decision must be either "option_A" or "option_B", and the instruction to run at the end must be "decide_option(user, Choice)." which shall return the final decision.
The Prolog programs must also:
- include all the facts mentioned in the NL description;
- not explain what's the correct option;
- not include any comments.

Start by writing the axioms, which are exactly the same in both the biased and unbiased versions. Axioms must be generic and not specific to any of the two options. Once the axioms are done, start writing the unbiased version. Then, write the biased version by minimally editing the unbiased version. Finally, write a short NL represention of the axioms, avoiding sections or sub-sections and finishing with a sentence like: "The best practice is: ...".

You output must adhere to this template:
## Axioms
```prolog
% code here
```

## Unbiased
```prolog
:- consult('axioms').
% rest of the code here
```

## Biased
```prolog
:- consult('axioms').
% rest of the code here
```

## NL Representation of Axioms
`...`

Below, you'll find an example of expected input and output.

# Example

# Input:

## Unbiased NL Dilemma
{unbiased_description_example_1}

## Biased NL Dilemma
{biased_description_example_1}


# Output:

## Axioms
```prolog
{prolog_axioms_example_1}
```

## Unbiased
```prolog
{unbiased_prolog_example_1}
```

## Biased
```prolog
{biased_prolog_example_1}
```

## NL Representation of Axioms
`{axioms_description_example_1}`
"""

PROLOG2TEXT_SYSTEM_INSTRUCTION = """Rewrite the Prolog Dilemma below in first-person natural language (NL). Below you'll find an example of dilemma in Prolog and of how I encoded it into NL. 

Follow these rules:
- Be short and do not mention any best practice or inference rule under any circumstances. 
- End with a question about whether to choose Option A or Option B, as in the example below.
- Respond with the NL-encoding only. Do not include any explanation, commentary, or formatting.

The NL description must adhere to this format:
{{Problem_Context}}. {{Goal_Description}}.
I have two options:  
- Option A: {{OptionA_Description}}. 
- Option B: {{OptionB_Description}}.
{{Question}}?

## Example Prolog: 
{dilemma_example}

## Example NL: 
{description_example}

## Prolog Dilemma: 
{dilemma_to_encode}"""

def create_cache(file_name, create_fn):
	print(f'Creating cache <{file_name}>..')
	result = create_fn()
	with open(file_name, 'wb') as f:
		pickle.dump(result, f)
	return result

def load_cache(file_name):
	if os.path.isfile(file_name):
		print(f'Loading cache <{file_name}>..')
		with open(file_name,'rb') as f:
			return pickle.load(f)
	return None

def load_or_create_cache(file_name, create_fn):
	result = load_cache(file_name)
	if result is None:
		result = create_cache(file_name, create_fn)
	return result

def get_cached_values(value_list, cache, fetch_fn, cache_name=None, key_fn=lambda x:x, empty_is_missing=True, transform_fn=None, **args):
	missing_values = tuple(
		q 
		for q in unique_everseen(filter(lambda x:x, value_list), key=key_fn) 
		if key_fn(q) not in cache or (empty_is_missing and not cache[key_fn(q)])
	)
	if len(missing_values) > 0:
		cache.update({
			key_fn(q): v
			for q,v in fetch_fn(missing_values)
		})
		if cache_name:
			create_cache(cache_name, lambda: cache)
	cached_values = [
		cache[key_fn(q)] if q else None 
		for q in value_list
	]
	if transform_fn:
		cached_values = list(map(transform_fn, cached_values))
	return cached_values

_loaded_caches = {}
def instruct_model(prompts, model='llama3.2', api_key=None, **kwargs):
	# if api_key == 'ollama':
	# 	return instruct_ollama_model(prompts, api_key=api_key, model=model, **kwargs)
	return instruct_openai_model(prompts, api_key=api_key, model=model, **kwargs)
			
def instruct_ollama_model(prompts, system_instructions=None, model='llama3.1', options=None, temperature=0.5, top_p=1, output_to_input_proportion=2, non_influential_prompt_size=0, cache_path='cache/', **args):
	max_tokens = 4096
	if options is None:
		# For Mistral: https://www.reddit.com/r/LocalLLaMA/comments/16v820a/mistral_7b_temperature_settings/
		options = { # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
			"seed": 42, # Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
			"num_predict": max_tokens, # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
			"top_k": 40, # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
			"top_p": 0.95, # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
			"temperature": 0.7, # The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
			"repeat_penalty": 1., # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
			"tfs_z": 1, # Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value (e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)
			"num_ctx": 2**13,  # Sets the size of the context window used to generate the next token. (Default: 2048)
			"repeat_last_n": 64, # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
			# "num_gpu": 0, # The number of layers to send to the GPU(s). Set to 0 to disable.
		}
	else:
		options = copy.deepcopy(options) # required to avoid side-effects
	options.update({
		"temperature": temperature,
		"top_p": top_p,
	})
	def fetch_fn(instruction_prompt):
		system_instruction, missing_prompt = instruction_prompt
		_options = copy.deepcopy(options) # required to avoid side-effects
		if _options.get("num_predict",-2) == -2:
			prompt_tokens = 2*(len(missing_prompt.split(' '))-non_influential_prompt_size)
			_options["num_predict"] = int(output_to_input_proportion*prompt_tokens)
		response = ollama.generate(
			model=model,
			prompt=missing_prompt,
			stream=False,
			options=_options,
			keep_alive='1h',
			system=system_instruction,
		)
		# print(missing_prompt, response['response'])
		# return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
		return instruction_prompt, response['response']
	def parallel_fetch_fn(missing_prompt_list):
		n_processes = 1
		with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,n_processes)) as executor:
			futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
			for future in tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to Ollama"):
				i,o=future.result()
				yield i,o
	os.makedirs(cache_path, exist_ok=True)
	ollama_cache_name = os.path.join(cache_path, f"_{model.replace('-','_')}_cache.pkl")
	if ollama_cache_name not in _loaded_caches:
		_loaded_caches[ollama_cache_name] = load_or_create_cache(ollama_cache_name, lambda: {})
	__ollama_cache = _loaded_caches[ollama_cache_name]
	cache_key = json.dumps(options,indent=4)
	return get_cached_values(
		list(zip(system_instructions if system_instructions else [None]*len(prompts), prompts)), 
		__ollama_cache, 
		parallel_fetch_fn, 
		# key_fn=lambda x: (x,model,n,temperature,top_p,frequency_penalty,presence_penalty), 
		key_fn=lambda x: (x,model,cache_key),  
		empty_is_missing=True,
		cache_name=ollama_cache_name,
		transform_fn=None if 'deepseek' not in model else (lambda x: x.split('</think>')[-1].strip() if x else None)
	)

def instruct_openai_model(prompts, system_instructions=None, api_key=None, base_url=None, model='gpt-4o-mini', n=1, temperature=1, top_p=1, frequency_penalty=0, presence_penalty=0, cache_path='cache/', parallelise=True, **kwargs):
	chatgpt_client = openai.OpenAI(api_key=api_key, base_url=base_url)
	max_tokens = None
	adjust_max_tokens = True
	if '32k' in model:
		max_tokens = 32768
	elif '16k' in model:
		max_tokens = 16385
	elif model=='gpt-4o' or 'preview' in model or 'turbo' in model:
		max_tokens = 4096 #128000
		adjust_max_tokens = False
	elif model.startswith('o1') or model.startswith('o3') or model.startswith('o4'):
		max_tokens = 2**16
		adjust_max_tokens = False
	if not max_tokens:
		if model.startswith('gpt-4'):
			max_tokens = 8192
		else:
			max_tokens = 4096
			adjust_max_tokens = False
	# print('max_tokens', max_tokens)
	def fetch_fn(instruction_prompt):
		system_instruction, missing_prompt = instruction_prompt
		if system_instruction:
			messages = [ 
				{"role": "system", "content": system_instruction},
			]
		else:
			messages = []
		messages += [ 
			{"role": "user", "content": missing_prompt} 
		]
		prompt_max_tokens = max_tokens
		if adjust_max_tokens:
			prompt_max_tokens -= int(3*len(missing_prompt.split(' \n')))
		if prompt_max_tokens < 1:
			return instruction_prompt, None
		try:
			if model.startswith("o"): # some params not available in reasoning models
				response = chatgpt_client.chat.completions.create(
					model=model,
					messages=messages,
					max_completion_tokens=prompt_max_tokens,
					n=n,
					stop=None,
					frequency_penalty=frequency_penalty, 
					presence_penalty=presence_penalty
				)
			else:
				response = chatgpt_client.chat.completions.create(
					model=model,
					messages=messages,
					max_tokens=prompt_max_tokens,
					n=n,
					stop=None,
					temperature=temperature,
					top_p=top_p,
					frequency_penalty=frequency_penalty, 
					presence_penalty=presence_penalty
				)
			# print(response.choices)
			result = [
				r.message.content.strip() 
				for r in response.choices 
				if r.message.content != 'Hello! It seems like your message might have been cut off. How can I assist you today?'
			]
			if len(result) == 1:
				result = result[0]
			return instruction_prompt, result # return also the missing_prompt otherwise asynchronous prompting will shuffle the outputs
		except Exception as e:
			print(f'OpenAI returned this error: {e}')
			return instruction_prompt, None
	def parallel_fetch_fn(missing_prompt_list):
		n_processes = multiprocessing.cpu_count() if parallelise else 1
		# Using ThreadPoolExecutor to run queries in parallel with tqdm for progress tracking
		with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,n_processes)) as executor:
			futures = [executor.submit(fetch_fn, prompt) for prompt in missing_prompt_list]
			for e,future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(missing_prompt_list), desc="Sending prompts to OpenAI")):
				i,o=future.result()
				yield i,o
	os.makedirs(cache_path, exist_ok=True)
	openai_cache_name = os.path.join(cache_path, f"_{model.replace('-','_')}_cache.pkl")
	if openai_cache_name not in _loaded_caches:
		_loaded_caches[openai_cache_name] = load_or_create_cache(openai_cache_name, lambda: {})
	__openai_cache = _loaded_caches[openai_cache_name]
	return get_cached_values(
		list(zip(system_instructions if system_instructions else [None]*len(prompts), prompts)), 
		__openai_cache, 
		parallel_fetch_fn, 
		# key_fn=lambda x: (x,model,n,temperature,top_p,frequency_penalty,presence_penalty), 
		key_fn=lambda x: (x,model,temperature,top_p,frequency_penalty,presence_penalty,n), 
		empty_is_missing=True,
		cache_name=openai_cache_name,
		transform_fn=None if 'deepseek' not in model else (lambda x: x.split('</think>')[-1].strip() if x else None)
	)

def get_document_list(directory):
	doc_list = []
	for obj in os.listdir(directory):
		obj_path = os.path.join(directory, obj)
		if os.path.isfile(obj_path):
			doc_list.append(obj_path)
		elif os.path.isdir(obj_path):
			doc_list.extend(get_document_list(obj_path))
	return doc_list

def run_prolog_script(file, profiler_path="./prolog_profiler_wrapper.pl", timeout=5):
	"""
	Run the Prolog JSON profiler wrapper on each file in `files`.
	Returns a list of dictionaries with profiling results.
	"""
	try:
		proc = subprocess.run(
			["swipl", "-q", "-s", profiler_path, "--", file],
			check=True,
			stdin=subprocess.DEVNULL,      # new
			close_fds=True,
			capture_output=True,
			text=True,
			timeout=timeout  # seconds
		)
		data = json.loads(proc.stdout)
		data["file"] = file
		return data
	except subprocess.TimeoutExpired as exc:
		# Python ≥ 3.11 exposes the underlying process via exc.process;
		# fall back to Popen for older versions.
		if hasattr(exc, "process") and exc.process:
			exc.process.kill()
			exc.process.communicate()      # flush & close pipes
		# print(f"[ERROR] Timeout expired while running Prolog script on {file}", file=sys.stderr)
		pass
	except subprocess.CalledProcessError as e:
		# print(f"[ERROR] {file}: {e.stderr.strip()}", file=sys.stderr)
		pass
	except json.JSONDecodeError as e:
		print(f"[ERROR] Failed to parse JSON for {file}: {e}", file=sys.stderr)
	return None

def run_prolog_script_from_content(content, profiler_path="./prolog_profiler_wrapper.pl", timeout=5):
	"""
	Save Prolog content to a temporary file and run the profiler on it.
	"""
	temp_path = None
	with tempfile.NamedTemporaryFile(suffix=".pl", mode="w", delete=False) as temp_file:
		temp_file.write(content)
		temp_file.flush()  # Ensure it's written
		temp_path = temp_file.name
		output = run_prolog_script(temp_path, profiler_path=profiler_path)
	if temp_path and os.path.exists(temp_path):
		try:
			os.remove(temp_path)
		except Exception as e:
			print(f"[WARNING] Failed to delete temporary file {temp_path}: {e}", file=sys.stderr)
	return output

def get_decision_and_explanation_from_output(gpai_output, bias):
	decision_pattern = r'[*#\s"\'()\n]*Decision[*#\s"\'()\n]*[:\n][*#\s"\'\n]*([^\n]+)[*#\s"\']*'
	explanation_pattern = r'[*#\s"\'()\n]*Explanation[*#\s"\'()\n]*[:\n][*#\s"\'\n]*([^\n]+)[*#\s"\']*'
	
	# Find all matches for the decision pattern
	cs_matches = list(re.finditer(decision_pattern, re.sub('Answer','Decision',re.sub('Option:','Decision:',gpai_output, re.IGNORECASE), re.IGNORECASE), re.IGNORECASE))
	# Extract the last match if it exists
	decision = cs_matches[-1].group(1).strip().strip('.*') if cs_matches else None
	# Find all matches for the explanation pattern
	se_matches = list(re.finditer(explanation_pattern, gpai_output, re.IGNORECASE))
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

def get_texts_similarity(text_1, texts):
	if not isinstance(texts, (list, tuple)):
		embeddings = SEMANTIC_SIMILARITY_MODEL.encode([text_1, texts])  # Returns a NumPy array
		return float(cosine_similarity(embeddings)[0][1])
	# encode all texts in one go: first element is text_1, then the list
	embeddings = SEMANTIC_SIMILARITY_MODEL.encode([text_1] + texts)
	# compute full pairwise cosine matrix
	sim_matrix = cosine_similarity(embeddings)
	# sim_matrix[0] is similarities of text_1 against everything,
	# so skip the zero‐th element (which is similarity with itself, i.e. 1.0)
	return list(map(float,sim_matrix[0, 1:].tolist()))

def get_texts_similarity_with_Levenshtein(text_1, text_2):
	# Calculate Levenshtein distance
	distance = Levenshtein.distance(text_1, text_2)
	
	# Normalize to a similarity score between 0 and 1
	max_len = max(len(text_1), len(text_2))
	if max_len == 0:
		return 1.0  # Both texts are empty, so they are identical
	similarity = 1 - (distance / max_len)
	return similarity

def get_index_set_of_duplicated_texts(texts, max_similarity_threshold=MAX_SIMILARITY_THRESHOLD_FOR_DILEMMAS):
	# Compute embeddings
	embeddings = SEMANTIC_SIMILARITY_MODEL.encode(texts)  # Returns a NumPy array

	# Compute pairwise cosine similarity matrix
	cosine_sim_matrix = cosine_similarity(embeddings)

	# Find index pairs with similarity above MAX_SIMILARITY_THRESHOLD_FOR_DILEMMAS (excluding self-pairs)
	n = len(texts)
	similar_pairs = []
	for i in range(n):
		for j in range(i + 1, n):  # Avoid duplicate and self-comparisons
			if cosine_sim_matrix[i, j] > max_similarity_threshold:
				similar_pairs.append((i, j, cosine_sim_matrix[i, j]))

	# Optionally print results
	for i, j, sim in similar_pairs:
		print(f"Texts {i} and {j} have similarity: {sim:.4f}")

	similarity_set_list = []
	for i, j, _ in similar_pairs:
		set_found = False
		for similarity_set in similarity_set_list:
			if i in similarity_set:
				similarity_set.add(j)
				set_found = True
			elif j in similarity_set:
				similarity_set.add(i)
				set_found = True
		if not set_found:
			similarity_set_list.append(set([i,j]))

	# similarity_set_list = list(map(sorted,similarity_set_list))

	duplicated_texts = set()
	for similarity_set in similarity_set_list:
		duplicated_texts.update(sorted(similarity_set)[1:])
	return duplicated_texts, similarity_set_list

def read_file_content(path):
	with open(path, "r", encoding="utf-8") as f:
		return f.read()


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

def clean_prolog_code(prolog_code):
	prolog_code = prolog_code.replace('decide_best_option','decide_option')
	# Remove single-line comments starting with %
	code_no_line_comments = re.sub(r'%.*', '', prolog_code)
	# Remove block comments /* ... */
	code_no_block_comments = re.sub(r'/\*.*?\*/', '', code_no_line_comments, flags=re.DOTALL)
	# Remove excess blank lines
	code_cleaned = re.sub(r'\n\s*\n+', '\n', code_no_block_comments)
	# Strip leading/trailing whitespace
	code_cleaned = code_cleaned.strip()
	# assert not prolog_code or code_cleaned, f"prolog_code: {prolog_code}\ncode_cleaned:{code_cleaned}"
	return code_cleaned

def extract_prolog_program_dict_list(text):
	pattern = re.compile(
		r"^.*?Axioms[^\n]*\n+[`]+prolog(?P<axioms>[^`]+)[`]+.*?"
		r"^.*?Unbiased[^\n]*\n+[`]+prolog(?P<unbiased>[^`]+)[`]+.*?"
		r"^.*?Biased[^\n]*\n+[`]+prolog(?P<biased>[^`]+)[`]+.*?"
		r"^.*?NL[^\n]*\n+[`]*(?P<axioms_description>[^\n]+)[`]*",
		re.DOTALL | re.MULTILINE
	)
	dilemma_pairs = []
	if text:
		for m in re.finditer(pattern, text):
			axioms, unbiased, biased, axioms_description = m.group('axioms').strip(), m.group('unbiased').strip(), m.group('biased').strip(), m.group('axioms_description').strip()
			if not axioms or not unbiased or not biased or not axioms_description:
				continue
			if not 'best practice is:' in axioms_description:
				# print(f'Cannot find best practice:')
				# print(axioms_description)
				# print('@'*10)
				# assert False
				continue
			dilemma_pairs.append({
				"axioms": clean_prolog_code(axioms),
				"axioms_description": axioms_description,
				"unbiased_prolog": clean_prolog_code(unbiased),
				"biased_prolog": clean_prolog_code(biased)
			})
	# assert dilemma_pairs, f"Cannot find dilemmas for text: {text}"
	# if not dilemma_pairs:
	# 	print(f'Cannot find Prolog dilemmas: {text}')
	# 	print('#'*10)
	return dilemma_pairs

get_prolog_program_content = lambda axioms, facts: ' '.join((axioms, facts.replace(":- consult('axioms').",''))) # we're gonna merge the axioms file with the facts file

def get_bias_validation_and_explanation_from_output(gpai_output):
	decision_pattern = r'[*#\s"\'()\n]*Decision[*#\s"\'()\n]*[:\n][*#\s"\'\n]*([^\n]+)[*#\s"\']*'
	explanation_pattern = r'[*#\s"\'()\n]*Explanation[*#\s"\'()\n]*[:\n][*#\s"\'\n]*([^\n]+)[*#\s"\']*'
	
	# Find all matches for the decision pattern
	cs_matches = list(re.finditer(decision_pattern, re.sub('Answer','Decision',re.sub('Option:','Decision:',gpai_output, re.IGNORECASE), re.IGNORECASE), re.IGNORECASE))
	# Extract the last match if it exists
	decision = cs_matches[-1].group(1).strip().strip('.*') if cs_matches else None
	# Find all matches for the explanation pattern
	se_matches = list(re.finditer(explanation_pattern, gpai_output, re.IGNORECASE))
	# Extract the last match if it exists
	explanation = se_matches[-1].group(1).strip().strip('.*') if se_matches else ''
	if not decision:
		return False, None
	if decision.casefold().startswith('Yes'.casefold()) or ('Yes'.casefold() in decision.casefold() and 'No'.casefold() not in decision.casefold()):
		decision = True
	else:
		decision = False
	return decision, explanation