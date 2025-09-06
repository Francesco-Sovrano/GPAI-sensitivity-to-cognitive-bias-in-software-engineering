# Replication Package — Mitigating Prompt‑Induced Cognitive Biases in GPAI for Software Engineering

This repository is the **replication package** for the paper *Mitigating Prompt‑Induced Cognitive Biases in General‑Purpose AI for Software Engineering*. It contains the datasets, experiment scripts, and plotting utilities needed to reproduce the results reported in the paper.

> **Dataset provenance.** The dilemmas and axiomatic backgrounds used here come from the **PROBE‑SWE** benchmark (dynamic SE dilemmas with biased/unbiased paired prompts). A dump is included under `./dataset/`.

---

## Contents

- `1_compute_bias_sensitivity.py` — run the core experiments for a given model and prompting strategy.
- `2_visualize_bias_sensitivity.py` — generate per‑bias plots, complexity‑tier breakdowns, and summary PDFs.
- `3_analyze_strategy_effectiveness.py` — aggregate and visualize per‑strategy effectiveness (heatmaps/boxplots).
- `lib.py` — shared utilities, the LLM client wrapper (OpenAI/Groq compatible), caching, and helpers.
- `dataset/` — augmented dilemmas for several base models (JSON).
- `generated_output_data/` — where result JSONs and figures are written.
- `logs/` — optional run logs when you pipe `stdout/stderr` there.
- `requirements.txt` — Python dependencies.
- `run_all_experiments.sh` — example batch runner (you may tailor it to your keys/models/settings).
- `LICENSE` — MIT.

---

## Prerequisites

- Python **3.10+**
- An API key for at least one provider
  - **OpenAI** (e.g., `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4.1-nano`)
  - **Groq** (e.g., `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `deepseek-r1-distill-llama-70b`)

Install dependencies in a fresh virtual environment:

```bash
python3 -m venv .env
source .env/bin/activate             # Windows: .env\Scripts\activate
pip install -r requirements.txt
```

---

## Configure API Keys

The scripts automatically pick the correct base URL from the chosen model name and will look for API keys in the environment:

- **OpenAI models** → read `OPENAI_API_KEY` and call `https://api.openai.com/v1`
- **Groq models** → read `GROQ_API_KEY` and call `https://api.groq.com/openai/v1`

### Option A — Export in your shell (recommended)

**macOS/Linux**

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

**Windows (PowerShell)**

```powershell
setx OPENAI_API_KEY "sk-..."
setx GROQ_API_KEY "gsk_..."
# Restart your shell so the variables are available
```

You can also prefix a single command:

```bash
OPENAI_API_KEY="sk-..." python 1_compute_bias_sensitivity.py --model gpt-4o-mini
```

### Option B — Source a local file

Create a file (e.g., `keys.env`) that contains:

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

Then source it before running:

```bash
source keys.env
```

> **Security note:** never commit your API keys. The sample shell script is provided for convenience—edit it locally and keep your keys private.

---

## Quick Start

Run the baseline experiment for a single model (5 independent runs per dilemma):

```bash
python 1_compute_bias_sensitivity.py   --model gpt-4o-mini   --n_independent_runs_per_task 5
```

Results are written to `generated_output_data/` as JSON files whose names encode all relevant flags.

### Choosing a provider/model

- **OpenAI examples**
  - `--model gpt-4o-mini`
  - `--model gpt-4.1-mini`
  - `--model gpt-4.1-nano`
- **Groq examples**
  - `--model llama-3.3-70b-versatile`
  - `--model llama-3.1-8b-instant`
  - `--model deepseek-r1-distill-llama-70b`

> The script routes OpenAI‑prefixed models to `OPENAI_API_KEY` and Groq‑hosted models to `GROQ_API_KEY` automatically.

### Evaluate prompting strategies (flags)

The main script exposes flags that mirror the strategies discussed in the paper:

| Strategy (paper) | What it does | Flag(s) |
|---|---|---|
| **Baseline (∅)** | No mitigation | *(no extra flags)* |
| **Chain‑of‑Thought (CoT)** | Ask the model to reason step‑by‑step | `--chain_of_thought` |
| **Implication Prompting (IMP)** | Ask for implications and why a decision might be biased | `--implication_prompting` |
| **Self‑Debiasing (BW)** | Add a bias‑warning instruction | `--bias_warning` |
| **Self‑Debiasing (IsD)** | Impersonated “unbiased engineer” instruction | `--impersonified_self_debiasing` |
| **sAX** | *Self axiomatic background elicitation* (extract + apply SE best‑practice cues in the explanation) | `--self_axioms_elicitation` |
| **2sAX** | *Two‑step axiomatic background elicitation* (extract cues first, then use them) | `--bistep_axioms_elicitation` |
| **ProbeAX** | Append option‑agnostic axiomatic backgrounds from PROBE-SWE as reasoning cues | `--inject_axioms` |

**Examples**

The examples below are ran on gpt-4o-mini (flag `--model`) on the whole dataset (flag `--data_model_list`).

∅ (no-strategy baseline):
```bash
python 1_compute_bias_sensitivity.py --model gpt-4o-mini --n_independent_runs_per_task 5 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b  
```

sAX + BW (recommended in the paper):
```bash
python 1_compute_bias_sensitivity.py --model gpt-4o-mini --self_axioms_elicitation --bias_warning --n_independent_runs_per_task 5 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b  
```

sAX + BW + IsD (recommended in the paper):
```bash
python 1_compute_bias_sensitivity.py --model gpt-4o-mini --self_axioms_elicitation --bias_warning --impersonified_self_debiasing --n_independent_runs_per_task 5 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b  
```

2sAX:
```bash
python 1_compute_bias_sensitivity.py --model gpt-4o-mini --bistep_axioms_elicitation --n_independent_runs_per_task 5 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b
```

Self‑debiasing (BW + IsD):
```bash
python 1_compute_bias_sensitivity.py --model gpt-4o-mini --bias_warning --impersonified_self_debiasing --n_independent_runs_per_task 5 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b
```

Chain‑of‑thought:
```bash
python 1_compute_bias_sensitivity.py --model gpt-4o-mini --chain_of_thought --n_independent_runs_per_task 5 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b
```

Implication prompting:
```bash
python 1_compute_bias_sensitivity.py --model gpt-4o-mini --implication_prompting --n_independent_runs_per_task 5 --data_model_list gpt-4.1-mini gpt-4o-mini deepseek-r1-distill-llama-70b
```

---

## Visualize and Summarize Results

After running experiments on all models, create the figures and summary PDFs (written to `generated_output_data/` by `2_visualize_bias_sensitivity.py` and to `strategy_effectiveness_analyses/` by `3_analyze_strategy_effectiveness`) using the scripts `2_visualize_bias_sensitivity.py` and `3_analyze_strategy_effectiveness`.

**Examples**

∅ (no-strategy baseline):
```bash
python 2_visualize_bias_sensitivity.py
```

sAX + BW (recommended in the paper):
```bash
python 2_visualize_bias_sensitivity.py --self_axioms_elicitation --bias_warning 
```

All strategies:
```bash
python 3_analyze_strategy_effectiveness --show_figures
```

---

## Reproducing the Paper (Workflow Recap)

1. **Set up** a clean virtualenv and install deps.
2. **Export API keys** for the providers you intend to use.
3. **Run experiments** with the desired strategies (baseline, sAX, 2sAX, CoT, self‑debiasing, IMP).
4. **Generate plots** with `2_visualize_bias_sensitivity.py` and **aggregate strategy results** with `3_analyze_strategy_effectiveness.py`.
5. Inspect PDFs in `generated_output_data/` and `strategy_effectiveness_analyses/`.

API usage incurs cost; consider lowering `--n_independent_runs_per_task` or running on fewer models when testing your setup.

---

## License

This project is released under the **MIT License** (see `LICENSE`).

---

## Troubleshooting

- *401/permission errors*: verify the correct key is exported for the selected model/provider.
- *Rate limits*: reduce parallelism (defaults are conservative) or try again later.