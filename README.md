# Replication Package — Mitigating Prompt‑Induced Cognitive Biases in GPAI for Software Engineering

This repository is the **replication package** for the paper *Mitigating Prompt‑Induced Cognitive Biases in General‑Purpose AI for Software Engineering*. It contains the scripts, data products, annotations, and documentation needed to inspect and reproduce the results reported in the paper across four parts of the study:

1. controlled paired-prompt experiments on PROBE-SWE-derived software-engineering dilemmas;
2. strategy-effectiveness aggregation and figure generation for RQ1 and RQ2;
3. thematic coding / lexicon-based analysis of GPAI system behaviours for RQ3;
4. open-ended qualitative analysis and post-hoc DevGPT validation for RQ4 and the real-world prompt audit.

A copy of the paper is included in this package as:
- [FSE_2026_Mitigating_Prompt_Induced_Cognitive_Biases_in_General_Purpose_AI_for_Software_Engineering.pdf](paper/FSE_2026_Mitigating_Prompt_Induced_Cognitive_Biases_in_General_Purpose_AI_for_Software_Engineering.pdf)


> **Dataset provenance.** The dilemmas and axiomatic backgrounds used for the main experiments come from the **PROBE‑SWE** benchmark (dynamic SE dilemmas with biased/unbiased paired prompts). A dump is included under `./dataset/`, and the original benchmark is available at: <https://github.com/Francesco-Sovrano/PROB-SWE>.
>
> The **DevGPT** post‑hoc analysis additionally relies on the external DevGPT dataset (not redistributed here); see the DevGPT section below for download instructions.

---

## Contents

Top-level files required for FSE artifact evaluation:

- `README.md` - overview, reproduction paths, and artifact location.
- `REQUIREMENTS` - hardware/software requirements.
- `INSTALL` - installation steps, smoke test, and expected outputs.
- `STATUS` - claimed badges and justification.
- `LICENSE` - distribution rights (MIT).
- `paper/FSE_2026_Mitigating_Prompt_Induced_Cognitive_Biases_in_General_Purpose_AI_for_Software_Engineering.pdf` - copy of the accepted paper.

Main research material:

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
- `devgpt_bias_features_analysis/` — post‑hoc audit of bias‑inducing *linguistic cues* in real‑world coding prompts from DevGPT (scripts + manual codebook + corrected labels; requires external DevGPT download).
- `thematic_coding_of_gpai_systems_behaviours/` — RQ3 lexicon/codebook + analysis script for thematic coding of GPAI behaviours; includes precomputed outputs under `bias_se_analysis/`.
- `open_ended_dilemma_qualitative_analysis/` — scripts and **manually_analyzed_data/** for the open‑ended dilemma qualitative analysis and associated figures.

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

The full controlled prompt experiments require:

- the benchmark JSON files in `dataset/` named `augmented_dilemmas_dataset_*.json`;
- API credentials for the selected model provider(s);
- internet access.

The intended full-run workflow is:

1. install the Python dependencies;
2. export `OPENAI_API_KEY` and/or `GROQ_API_KEY`;
3. run `1_compute_bias_sensitivity.py` with the desired strategy flags;
4. run `2_visualize_bias_sensitivity.py` and `3_analyze_strategy_effectiveness.py`;
5. optionally re-run RQ3, RQ4, and DevGPT analyses.

See `INSTALL` for concrete commands.

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

## Additional Artifacts and Analyses

### Post-hoc validation on real-world prompts (DevGPT)

The directory `devgpt_bias_features_analysis/` contains the scripts and **manual correction artifacts** used in the paper’s post-hoc validation on DevGPT (real-world developer–ChatGPT conversations).

**What’s in the folder**
- `analyze_bias_features_devgpt.py` — end-to-end pipeline (triage → coding-only filter → cue-type assignment → prevalence tables).
- `classify_devgpt_with_groq.py` — helper script to (re-)run the *coding-related* triage and cue classification with an OpenAI-compatible API.
- `DevGPT_manual_cue_codebook.md` — **manual cue codebook** (inclusion/exclusion criteria, label set, decision rules).
- `manually_corrected_entries.csv` — manually audited subset (final label column: `bias_cue_type_llm_corrected`).

#### DevGPT data dependency (download)
DevGPT is not redistributed in this replication package. Download `DevGPT.zip` from Zenodo (Version v9 used in the paper):

- https://zenodo.org/records/10086809  (DOI: 10.5281/zenodo.10086809)

You can keep the dataset **as a zip**; the pipeline accepts either a `.zip` file or an extracted directory.

#### Run (example)
```bash
python devgpt_bias_features_analysis/analyze_bias_features_devgpt.py \
  --probe ./dataset \
  --devgpt /path/to/DevGPT.zip \
  --out_dir devgpt_bias_features_analysis/out
```

Notes:
- The default `--ai_method hf_clf` fine-tunes a DeBERTa classifier and may benefit from a GPU (`--device cuda`).  
- First run may download HF models; cache locations can be customized via `HF_PRED_CACHE_DIR` / `SBERT_EMB_CACHE_DIR` (see script header).

---

### RQ3 thematic coding of GPAI system behaviours (lexicon-based)

The directory `thematic_coding_of_gpai_systems_behaviours/` contains:
- `codebook.md` — the **lexicon/codebook** (feature groups and matching rules).
- `thematic_analysis.py` — analysis script that counts lexicon features in model explanations and tests for differences.
- `bias_se_analysis/` — **precomputed** outputs (CSVs + PDFs) included for convenience.

#### Re-run from your own experiment outputs
1) Run `1_compute_bias_sensitivity.py` (baseline and/or mitigated strategies) to generate CSVs in `generated_output_data/`:
- `generated_output_data/1_llm_outputs_*.csv`

2) Zip one or more of those CSVs into `to_analyze.zip`:
```bash
cd generated_output_data
zip to_analyze.zip 1_llm_outputs_*.csv
cd ..
```

3) Run the thematic analysis:
```bash
python thematic_coding_of_gpai_systems_behaviours/thematic_analysis.py \
  --zip generated_output_data/to_analyze.zip \
  --output_path thematic_coding_of_gpai_systems_behaviours/bias_se_analysis
```

The input CSVs must contain (at least) these columns:
`bias_name`, `prompt_with_bias`, `decision_explanation_with_bias`, `sensitive_to_bias`.

---

### Open-ended dilemma qualitative analysis (manual coding)

The directory `open_ended_dilemma_qualitative_analysis/` contains the scripts and the **manually_analyzed_data/** used for the paper’s open-ended dilemma audit.

**Manual data (included)**
- `manually_analyzed_data/p1/` and `manually_analyzed_data/p2/` — two independent coding passes.
- `manually_analyzed_data/resolved_conflict/` — final resolved labels (used for plotting/stats).
- `manually_analyzed_data/combined_agreement_stats.csv`, `paper_stats.csv` — supporting stats/materials.

**Reproduce the summary figures**
```bash
cd open_ended_dilemma_qualitative_analysis
pip install -r requirements.txt
python visualize_results.py \
  --gpt_csv  manually_analyzed_data/resolved_conflict/extracted_gpt4o_checked.csv \
  --llama_csv manually_analyzed_data/resolved_conflict/extracted_llama_checked.csv \
  --out_dir data_visualization
```

The `data_merger.py` helper shows how the *candidates to manually audit* were derived from the CSV outputs of `1_compute_bias_sensitivity.py` (by intersecting cases where the baseline is sensitive and the mitigated setting is not).

## Citation

If you use this package, please cite the paper and acknowledge the *Mitigating Prompt‑Induced Cognitive Biases in General‑Purpose AI for Software Engineering* paper:
```text
@article{sovrano2026mitigating,
  title={Mitigating Prompt-Induced Cognitive Biases in General-Purpose AI for Software Engineering},
  author={Sovrano, Francesco and Dominici, Gabriele and Bacchelli, Alberto},
  journal={Proceedings of the ACM on Software Engineering},
  year={2026}
}
```

As well as the [PROBE-SWE](https://github.com/Francesco-Sovrano/PROB-SWE) benchmark :
```text
@article{sovrano2025general,
  title={Is General-Purpose AI Reasoning Sensitive to Data-Induced Cognitive Biases? Dynamic Benchmarking on Typical Software Engineering Dilemmas},
  author={Sovrano, Francesco and Dominici, Gabriele and Sevastjanova, Rita and Stramiglio, Alessandra and Bacchelli, Alberto},
  journal      = {CoRR},
  url          = {https://doi.org/10.48550/arXiv.2508.11278},
  doi          = {10.48550/ARXIV.2508.11278},
  year={2025}
}
```

---

## License

This artefact is released under the **MIT License** (see `LICENSE`).

---

## Troubleshooting

- *401/permission errors*: verify the correct key is exported for the selected model/provider.
- *Rate limits*: reduce parallelism (defaults are conservative) or try again later.