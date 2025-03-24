# Evaluating and Mitigating Prompt-Induced Cognitive Biases in General-Purpose AI for Software Engineering

Welcome to the replication package for the ICSE 2025 paper titled "Evaluating and Mitigating Prompt-Induced Cognitive Biases in General-Purpose AI for Software Engineering".

## Abstract

Cognitive biases are mental shortcuts that reduce effort but can distort judgment. In software engineering, biases like confirmation bias can cause developers to test their programs only with data that aligns with how they expect the program to work, rather than using data that might reveal inconsistencies or errors. Meanwhile, general-purpose AI (GPAI) is rapidly gaining traction in software engineering tasks, from code completion to architectural design, increasingly performing tasks traditionally done by developers and raising concerns that these AI systems may inherit or even amplify the biases present in their human-generated training data. This paper investigates prompt-induced biases in GPAI systems for software engineering tasks (e.g., GPT-4o, DeepSeek, LLama3.3) to determine if they are affected by the same cognitive biases observed in human decision-making. We focus on common harmful biases (e.g., availability bias, hyperbolic discounting, confirmation bias) by designing paired task instances, one negatively biased (nudging the GPAI toward a less agreeable answer) and one not. Our within-model evaluation shows that all tested GPAI systems are significantly sensitive (𝑝 < 0.05; 1-𝛽 ≥ 0.8) to at least five biases. Although cognitive biases may expedite human reasoning, their presence in GPAI systems can lead to suboptimal decisions, potentially resulting in flawed designs and inefficient resource allocation. Therefore, we explore mitigation strategies such as bias-aware chain-of-thought and we create a novel approach to counter bias, based on human-centred explainable AI techniques to create task-specific reasoning cues. Our results indicate that our approach reduces cognitive bias effects in nearly all advanced GPAI systems, except for the bandwagon effect. These findings shed light on how cognitive biases affect AI decision-making in software engineering and offer actionable recommendations to both practitioners and researchers for developing more robust, bias-resistant, and trustworthy AI systems.

## Overview
This repository contains resources and scripts for analyzing and visualizing the sensitivity of General-Purpose AI (GPAI) models to various cognitive biases within software engineering contexts. Specifically, it investigates how prompting strategies and bias scenarios affect GPAI decision-making.

## Repository Structure
- `gpai_study/`: Contains scripts for conducting experiments and analyzing data.
  - Scripts include:
    - Data collection and preprocessing (`0_get_stats_about_scenarios.py`, `1_get_decisions.py`, `1_get_decisions_by_persona.py`).
    - Analysis scripts (`2_visualize_results.py`, `3_chi_squared_analysis.py`, `4_global_sensitivity_analysis.py`).
    - `lib.py`: Shared utilities.
    - `run_all_experiments.sh`: Automated execution of all experiments.
    - `requirements.txt`: Python dependencies required for execution.
- `scenarios/`: Text-based scenario descriptions categorized by cognitive biases with unbiased and biased versions.
  - Types of cognitive biases covered:
    - Confirmation Bias
    - Hyperbolic Discounting
    - Availability Bias
    - Anchoring Bias
    - Overconfidence Bias
    - Bandwagon Effect
    - Framing Effect
    - Hindsight Bias

## Scenario Types
- `_1st_person_cued_`: First-person scenarios with explicit cues designed to evoke cognitive biases.
- `_1st_person_shallow_`: First-person scenarios without explicit bias cues (baseline scenarios).

## Environment Setup

Ensure Python dependencies are installed:
```bash
pip install -r requirements.txt
```

### System Specifications

This repository is tested and recommended on:

- OS: Linux (Debian 5.10.179 or newer) and macOS (15.3.1 Sequoia or newer)
- Python version: 3.9 or newer

### Installation of OpenAI Keys

To run the experiments with OpenAI's GPT models, you must set up two environment variables: `OPENAI_ORGANIZATION` and `OPENAI_API_KEY`. These variables represent your OpenAI organization identifier and your API key respectively.

On UNIX-like Operating Systems (Linux, MacOS):
1. Open your terminal.
2. To set the `OPENAI_ORGANIZATION` variable, run:
   ```bash
   export OPENAI_ORGANIZATION='your_organization_id'
   ```
3. To set the `OPENAI_API_KEY` variable, run:
   ```bash
   export OPENAI_API_KEY='your_api_key'
   ```
4. These commands will set the environment variables for your current session. If you want to make them permanent, you can add the above lines to your shell profile (`~/.bashrc`, `~/.bash_profile`, `~/.zshrc`, etc.)

To ensure you've set up the environment variables correctly:

1. In your terminal or command prompt, run:
   ```bash
   echo $OPENAI_ORGANIZATION
   ```
   This should display your organization ID.
   
2. Similarly, verify the API key:
   ```bash
   echo $OPENAI_API_KEY
   ```

Ensure that both values match what you've set.

### Installation of Groq API Keys

To run the experiments with DeepSeek and LLama 3.3, you must also set up one environment variable: `GROQ_API_KEY`. This variable represents your [Groq](https://groq.com) API key.

On UNIX-like Operating Systems (Linux, macOS):

1. Open your terminal.
2. To set the `GROQ_API_KEY` variable, run:
   ```bash
   export GROQ_API_KEY='your_api_key'
   ```
3. This command will set the environment variable for your current session. If you want to make it permanent, you can add the above line to your shell profile (e.g., `~/.bashrc`, `~/.bash_profile`, `~/.zshrc`, etc.).

To ensure you've set up the environment variables correctly:

1. In your terminal or command prompt, run:
   ```bash
   echo $GROQ_API_KEY
   ```

Make sure that the value matches what you've set.

## Running the Experiments

To run all experiments, execute:
```bash
sh run_all_experiments.sh
```

## Results and Analysis

Generated results and sensitivity analyses are stored in `gpai_study/results/`, including summary tables (`.csv`) and visualizations (`.pdf`). Here you will find:

- **Summary Metrics:**
  - **Text Files:**  
    - `averages_1st_person_cued_.txt`  
    - `averages_1st_person_shallow_.txt`  
    These files provide aggregated metrics (such as average sensitivity scores) across different bias scenarios for each experimental condition.
  
- **Detailed Decision Logs:**
  - **CSV Files in the `_1st_person_cued_` folder:**
    - `sensitivity_scores_cued.csv`: Contains computed sensitivity scores for first-person cued scenarios.
    - `decisions_model-gpt-4o-2024-08-06_temperature-1_top_p-1.csv`: Detailed decision outputs for the GPT-4 model using specific experimental parameters.
    - `decisions_model-gpt-4o-mini-2024-07-18_temperature-1_top_p-1.csv`: Similar decision logs for a GPT-4 mini configuration.

- **Visualizations:**
  - Within several subdirectories (e.g., `_1st_person_shallow_CoT_`, `_1st_person_shallow_warning_`, and `_1st_person_cued_`), you will find a **figures** folder containing visual outputs:
    - **PDF Files:** Visualizations such as stacked bar charts (e.g., `sensitivity_stacked_correct_incorrect.pdf`) that compare biased versus unbiased task performance across scenarios.

These results help evaluate:
- **Performance Differences:** How AI decisions vary when presented with biased versus unbiased scenarios.
- **Sensitivity Scores:** Quantitative assessments of the AI model’s sensitivity to various cognitive biases.
- **Comparative Analyses:** Insights into how different prompting strategies affect decision-making.

## Contributing
Contributions to enhance scenarios, analysis methods, or general improvements are welcome. Please submit pull requests or open issues for discussions.
