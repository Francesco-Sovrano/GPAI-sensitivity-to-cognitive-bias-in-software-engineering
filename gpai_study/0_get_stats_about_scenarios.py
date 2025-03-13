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
	"--reasoning",
	type=str,
	required=True,
	help="Specify the format (e.g., 'cued', 'shallow')"
)
parser.add_argument(
	"--format",
	type=str,
	default='1st_person',
	help="Specify the format: '1st_person', '2nd_person'"
)
args = parser.parse_args()

scenarios_dir = os.path.join('../scenarios', f"_{args.format}_{args.reasoning}_")

# We assume that the top-level extracted folder (scenarios_dir) contains one directory per bias.
bias_dirs = [d for d in os.listdir(scenarios_dir) if os.path.isdir(os.path.join(scenarios_dir, d))]
stats = []
for bias in bias_dirs:
	bias_path = os.path.join(scenarios_dir, bias)
	# Assume two decision-making tasks per bias
	task_dirs = [d for d in os.listdir(bias_path) if os.path.isdir(os.path.join(bias_path, d))]
	
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
		
		stats.append({
			'type': 'biased',
			'n_chars': len(vignette_biased),
			'n_words': len(vignette_biased.split(' ')),
		})

		stats.append({
			'type': 'nonbiased',
			'n_chars': len(vignette_nonbiased),
			'n_words': len(vignette_nonbiased.split(' ')),
		})
		
# Compute averages for each category
def compute_averages(stat_list):
	total_chars = sum(item['n_chars'] for item in stat_list)
	total_words = sum(item['n_words'] for item in stat_list)
	count = len(stat_list)
	return total_chars / count if count > 0 else 0, total_words / count if count > 0 else 0

biased_stats = [s for s in stats if s['type'] == 'biased']
nonbiased_stats = [s for s in stats if s['type'] == 'nonbiased']

avg_biased_chars, avg_biased_words = compute_averages(biased_stats)
avg_nonbiased_chars, avg_nonbiased_words = compute_averages(nonbiased_stats)
avg_all_chars, avg_all_words = compute_averages(stats)

# Prepare output string
output = (
	f"Average Character and Word Counts\n"
	f"----------------------------------\n"
	f"Biased:\n  Average characters: {avg_biased_chars:.2f}\n  Average words: {avg_biased_words:.2f}\n\n"
	f"Nonbiased:\n  Average characters: {avg_nonbiased_chars:.2f}\n  Average words: {avg_nonbiased_words:.2f}\n\n"
	f"Overall:\n  Average characters: {avg_all_chars:.2f}\n  Average words: {avg_all_words:.2f}\n"
)

# Ensure results directory exists
results_dir = "./results"
if not os.path.exists(results_dir):
	os.makedirs(results_dir)

# Save the output to a text file
output_file = os.path.join(results_dir, f"averages_{args.format}_{args.reasoning}_.txt")
with open(output_file, "w") as f:
	f.write(output)

print("Averages computed and saved to", output_file)