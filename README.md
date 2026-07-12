# CONTEST: CONtraction TEST for LLM Uncertainty Decomposition

Code for the paper **"Contested, not Confused: Why LLM Sarcasm Uncertainty Cannot Separate Human Disagreement from Model Ignorance"** (under review).

---

## Overview

CONTEST is a disagreement-grounded diagnostic framework that tests whether LLM uncertainty signals can separate epistemic (model-ignorance) errors from aleatoric (human-irreducibly-split) errors in sarcasm / irony detection.

Three experimental layers:

| Layer | Script | Description |
|---|---|---|
| E1 | `run_e1.py` | LLM error rates vs. human disagreement tercile (frontier models) |
| E2 | `run_e2_delta_para.py` | Paraphrase-cosine delta vs. LLM error correlation |
| E3 analysis | `run_e3_analyze.py` | Asymmetric contraction ΔAsym over Qwen2.5 0.5B→32B capacity sweep |
| E3 cross-family API | `run_e3_xfamily_api.py` | Cross-family sweep (Qwen3/Gemma3/Llama-3.x) via OpenRouter |
| E3 cross-family analysis | `run_e3_xfamily_analyze.py` | ΔAsym for cross-family results |
| E3 hate speech API | `run_e3_mhs_api.py` | MHS hate speech generalisation sweep |
| E3 hate speech analysis | `run_e3_mhs_analyze.py` | ΔAsym for MHS |
| E4/E5 method | `run_e3_method.py` | AUROC + AUCAC for contraction probe vs. baselines |
| Distribution metrics | `run_metareview_distmetrics.py` | Brier / JSD re-scoring against human label distributions |
| Cost accounting | `run_metareview_costaccounting.py` | Honest cost-accuracy frontier for cascade routing |
| Learned probe | `run_metareview_classifier.py` | 10-seed CV AUROC for learned contraction probe |
| Strong baselines | `run_strong_baselines.py` | AUROC/AUCAC for SVM/LSTM/BERT/RoBERTa/deep-ensemble baselines |
| Hate E1+E2 | `run_hate.py` | E1+E2 on MHS hate speech corpus |

Visualisation scripts: `plot_capacity_curves.py`, `plot_costacc.py`, `plot_distmetrics.py`, `plot_distmetrics_meta.py`, `plot_strong_baselines.py`, `plot_hate_results.py`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` in this directory and fill in your keys:

```bash
cp .env.example .env
# then edit .env
```

The `.env` file is loaded at runtime; keys are never printed to stdout.

### 3. Obtain corpora

The three perspectivist irony corpora used:

| Corpus | Reference | Access |
|---|---|---|
| CSC (Controversial Sarcasm Corpus) | Jang et al. 2024 | Contact authors |
| MultiPICo | Casola et al. 2024 | [GitHub](https://github.com/dhfbk/multipico) |
| EPIC | Frenda et al. 2023 | [GitHub](https://github.com/Humor-Research/EPIC) |
| MHS (Measuring Hate Speech) | Kennedy et al. 2022 | [HuggingFace](https://huggingface.co/datasets/ucberkeley-dsp/measuring-hate-speech) |

Place aggregated parquets under `data/raw/` relative to the **project root** (one level above `code_release/`). The expected schema for each is documented inside the respective run script.

---

## Directory layout

Scripts must be run from the **project root** (parent of `code_release/`), or the path constants in each script (`BASE = Path(__file__).parent.parent`) automatically resolve correctly.

```
<project_root>/
├── code_release/           # this directory
│   ├── run_e1.py
│   ├── ...
│   ├── requirements.txt
│   └── .env                # your API keys (not committed)
├── data/
│   ├── raw/                # original corpora (not redistributed)
│   └── processed/
└── results/
    ├── sarcasm/            # aggregated parquets
    ├── llm_e1/             # E1 outputs
    ├── llm_e2/             # E2 outputs
    ├── llm_e3/             # E3 outputs + CSVs
    ├── llm_hate/           # MHS outputs
    ├── metareview/         # distribution metrics, cost accounting, learned probe
    └── strong_baselines/   # strong baseline results
```

Figures are saved to `<project_root>/figures/`.

---

## Reproducing main results

### E1 (frontier models, sarcasm)

```bash
python code_release/run_e1.py --n 400 --phase 1 \
    --datasets CSC,MultiPICo,EPIC --providers gpt,claude \
    --gpt_model gpt-4o-mini --claude_model anthropic/claude-haiku-4.5
```

Frontier models (`--gpt_client openrouter`):
```bash
python code_release/run_e1.py --n 400 --phase frontier400 \
    --gpt_model openai/gpt-4.1 --claude_model anthropic/claude-opus-4.8 \
    --gpt_client openrouter
```

### E3 analysis (Qwen2.5 sweep — requires pre-run CSVs)

Capacity-sweep CSVs (`e3_q0p5b_CSC.csv`, ..., `e3_q32b_EPIC.csv`) are not included in the repository. The E3 GPU sweep was run on a university cluster; the resulting CSVs must be placed in `results/llm_e3/`.

```bash
python code_release/run_e3_analyze.py   # reads results/llm_e3/e3_q*.csv
```

### E3 cross-family (via OpenRouter API)

```bash
python code_release/run_e3_xfamily_api.py \
    --families qwen3,gemma3,llama3x \
    --corpora CSC,MultiPICo,EPIC
python code_release/run_e3_xfamily_analyze.py
```

### E4/E5 method (AUROC / AUCAC)

```bash
python code_release/run_e3_method.py
python code_release/run_metareview_costaccounting.py
```

### Strong baselines

```bash
python code_release/run_strong_baselines.py
```

---

## Reproducibility notes

- All random seeds are set to `seed=42`.
- Bootstrap tests use 4000 resamples (main sweep) or 2000 resamples (AUROC CI).
- The prediction threshold for binary classification is `p_yes >= 0.5` (inclusive).
- Human disagreement terciles use a rank-based split: `np.digitize(rankdata(dis_mi)/(n+1), [1/3, 2/3])`. This differs from quantile-based splits for corpora with discrete dis_mi values (e.g., EPIC).

---

## Citation

If you use this code, please cite:

```
[citation to be added upon publication]
```

---

## License

MIT License. See LICENSE file (to be added upon publication).
