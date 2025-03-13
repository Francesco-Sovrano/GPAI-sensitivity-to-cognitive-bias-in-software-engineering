# Evaluating and Mitigating Prompt-Induced Cognitive Biases in General-Purpose AI for Software Engineering

Welcome to the replication package for the ICSE 2025 paper titled "Evaluating and Mitigating Prompt-Induced Cognitive Biases in General-Purpose AI for Software Engineering".

## Abstract


## Repository Contents


## System Specifications

This repository is tested and recommended on:

- OS: Linux (Debian 5.10.179 or newer) and macOS (15.3.1 Sequoia or newer)
- Python version: 3.9 or newer


## Environment Setup


## Installation of OpenAI Keys

To use this package, you must set up two environment variables: `OPENAI_ORGANIZATION` and `OPENAI_API_KEY`. These variables represent your OpenAI organization identifier and your API key respectively.

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


## Run the AI Experiments


## User Study Interface and Results


