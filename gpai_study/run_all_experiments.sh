#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

python3.9 -m venv .env
source .env/bin/activate

python 0_get_stats_about_scenarios.py --reasoning cued
python 0_get_stats_about_scenarios.py --reasoning shallow

#################################
### Shallow Reasoning
#################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow # 32B

python 2_visualize_results.py --reasoning shallow #--show_figures

python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow

######################################
### Shallow Reasoning + Bias Warning
######################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow --bias_warning_in_system_instruction
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow --bias_warning_in_system_instruction
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --bias_warning_in_system_instruction
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow --bias_warning_in_system_instruction # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --bias_warning_in_system_instruction # 32B

python 2_visualize_results.py --reasoning shallow --bias_warning_in_system_instruction #--show_figures

python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow --bias_warning_in_system_instruction # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --bias_warning_in_system_instruction # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --bias_warning_in_system_instruction
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow --bias_warning_in_system_instruction
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow --bias_warning_in_system_instruction

######################################
### Shallow Reasoning + CoT (see: https://learn.microsoft.com/en-us/dotnet/ai/conceptual/chain-of-thought-prompting)
######################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow --chain_of_thought
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow --chain_of_thought
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --chain_of_thought
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow --chain_of_thought # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --chain_of_thought # 32B

python 2_visualize_results.py --reasoning shallow --chain_of_thought #--show_figures

python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow --chain_of_thought # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --chain_of_thought # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --chain_of_thought
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow --chain_of_thought
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow --chain_of_thought

######################################
### Shallow Reasoning + Impersonified Self-Debiasing
######################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow --impersonified_self_debiasing
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow --impersonified_self_debiasing
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --impersonified_self_debiasing
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow --impersonified_self_debiasing # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --impersonified_self_debiasing # 32B

python 2_visualize_results.py --reasoning shallow --impersonified_self_debiasing #--show_figures

python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow --impersonified_self_debiasing # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --impersonified_self_debiasing # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --impersonified_self_debiasing
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow --impersonified_self_debiasing
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow --impersonified_self_debiasing

######################################
### Shallow Reasoning + Implication Prompting
######################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow --implication_prompting
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow --implication_prompting
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --implication_prompting
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow --implication_prompting # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --implication_prompting # 32B

python 2_visualize_results.py --reasoning shallow --implication_prompting #--show_figures

python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow --implication_prompting # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --implication_prompting # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --implication_prompting
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow --implication_prompting
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow --implication_prompting

#################################
### Manually Cued Reasoning
#################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning cued
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning cued 
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning cued
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning cued # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning cued # 32B

python 2_visualize_results.py --reasoning cued #--show_figures

python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning cued # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning cued # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning cued
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning cued
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning cued

#####

python 4_global_sensitivity_analysis.py