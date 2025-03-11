#!/bin/bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

python3.9 -m venv .env
source .env/bin/activate

#################################
### Shallow Reasoning
#################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow &
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow &
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow &
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow # 32B
# python 1_get_decisions.py --model llama3.1 --reasoning shallow # 8B
# python 1_get_decisions.py --model llama3.2 --reasoning shallow # 3B

python 2_visualize_results.py --reasoning shallow #--show_figures

# python 3_chi_squared_analysis.py --model llama3.1 --reasoning shallow # 8B
# python 3_chi_squared_analysis.py --model llama3.2 --reasoning shallow # 3B
python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow

######################################
### Shallow Reasoning + Bias Warning
######################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow --bias_warning_in_system_instruction &
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow --bias_warning_in_system_instruction &
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --bias_warning_in_system_instruction &
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow --bias_warning_in_system_instruction # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --bias_warning_in_system_instruction # 32B
# python 1_get_decisions.py --model llama3.1 --reasoning shallow --bias_warning_in_system_instruction # 8B
# python 1_get_decisions.py --model llama3.2 --reasoning shallow --bias_warning_in_system_instruction # 3B

python 2_visualize_results.py --reasoning shallow --bias_warning_in_system_instruction #--show_figures

# python 3_chi_squared_analysis.py --model llama3.1 --reasoning shallow --bias_warning_in_system_instruction # 8B
# python 3_chi_squared_analysis.py --model llama3.2 --reasoning shallow --bias_warning_in_system_instruction # 3B
python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow --bias_warning_in_system_instruction # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --bias_warning_in_system_instruction # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --bias_warning_in_system_instruction
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow --bias_warning_in_system_instruction
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow --bias_warning_in_system_instruction

######################################
### Shallow Reasoning + Bias Warning + CoT (see: https://learn.microsoft.com/en-us/dotnet/ai/conceptual/chain-of-thought-prompting)
######################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought &
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought &
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought &
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 32B
# python 1_get_decisions.py --model llama3.1 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 8B
# python 1_get_decisions.py --model llama3.2 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 3B

python 2_visualize_results.py --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought #--show_figures

# python 3_chi_squared_analysis.py --model llama3.1 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 8B
# python 3_chi_squared_analysis.py --model llama3.2 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 3B
python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning shallow --bias_warning_in_system_instruction --chain_of_thought

#################################
### Cued Reasoning
#################################
python 1_get_decisions.py --model gpt-3.5-turbo-0125 --reasoning cued &
python 1_get_decisions.py --model gpt-4o-mini-2024-07-18 --reasoning cued 
python 1_get_decisions.py --model gpt-4o-2024-08-06 --reasoning cued &
python 1_get_decisions.py --model llama-3.3-70b-versatile --reasoning cued & # 70B
python 1_get_decisions.py --model deepseek-r1-distill-qwen-32b --reasoning cued # 32B
# python 1_get_decisions.py --model llama3.1 --reasoning cued # 8B
# python 1_get_decisions.py --model llama3.2 --reasoning cued # 3B

python 2_visualize_results.py --reasoning cued #--show_figures

# python 3_chi_squared_analysis.py --model llama3.1 --reasoning cued # 8B
# python 3_chi_squared_analysis.py --model llama3.2 --reasoning cued # 3B
python 3_chi_squared_analysis.py --model llama-3.3-70b-versatile --reasoning cued # 70B
python 3_chi_squared_analysis.py --model deepseek-r1-distill-qwen-32b --reasoning cued # 32B
python 3_chi_squared_analysis.py --model gpt-4o-mini-2024-07-18 --reasoning cued
python 3_chi_squared_analysis.py --model gpt-3.5-turbo-0125 --reasoning cued
python 3_chi_squared_analysis.py --model gpt-4o-2024-08-06 --reasoning cued
