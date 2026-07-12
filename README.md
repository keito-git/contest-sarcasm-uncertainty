# CONTEST: CONtraction TEST for LLM Uncertainty Decomposition

Code for the paper **"Contested, not Confused: Why LLM Sarcasm Uncertainty Cannot Separate Human Disagreement from Model Ignorance"** (under review).

---

## Overview

CONTEST is a disagreement-grounded diagnostic framework that tests whether LLM uncertainty signals can separate epistemic (model-ignorance) errors from aleatoric (human-irreducibly-split) errors in sarcasm / irony detection.

The scripts are grouped by role into three directories:

- **`experiments/`** — call LLMs / APIs to produce raw prediction outputs.
- **`analysis/`** — post-process the raw outputs into statistics, tables, and metrics.
- **`plotting/`** — render the figures used in the paper.

### `experiments/`

| Script | Description |
|---|---|
| `run_e1.py` | E1: LLM error rates vs. human-disagreement tercile (frontier models) |
| `run_e2_delta_para.py` | E2: paraphrase-cosine / sentiment-flip delta vs. LLM error correlation |
| `run_e3_xfamily_api.py` | E3 cross-family capacity sweep (Qwen3 / Gemma3 / Llama-3.x) via OpenRouter |
| `run_e3_mhs_api.py` | E3 hate-speech (MHS) generalisation sweep |
| `run_hate.py` | E1 + E2 on the MHS hate-speech corpus |

### `analysis/`

| Script | Description |
|---|---|
| `run_e3_analyze.py` | Asymmetric contraction ΔAsym over the Qwen2.5 0.5B→32B sweep |
| `run_e3_method.py` | AUROC + AUCAC for the contraction probe vs. baselines |
| `run_e3_xfamily_analyze.py` | ΔAsym for the cross-family results |
| `run_e3_mhs_analyze.py` | ΔAsym for MHS |
| `run_strong_baselines.py` | AUROC / AUCAC for strong (supervised / ensemble) baselines |
| `run_metareview_distmetrics.py` | Brier / JSD re-scoring against human label distributions |
| `run_metareview_costaccounting.py` | Honest cost-accuracy frontier for cascade routing |
| `run_metareview_classifier.py` | 10-seed CV AUROC for the learned contraction probe |

### `plotting/`

`plot_capacity_curves.py`, `plot_costacc.py`, `plot_distmetrics.py`, `plot_distmetrics_meta.py`, `plot_strong_baselines.py`, `plot_hate_results.py`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` in the **repository root** (this directory) and fill in your keys:

```bash
cp .env.example .env
# then edit .env
```

The `.env` file is loaded at runtime; keys are never printed to stdout.

### 3. Obtain corpora

The perspectivist irony / hate-speech corpora used:

| Corpus | Reference | Access |
|---|---|---|
| CSC (Conversational Sarcasm Corpus) | Jang & Frassinelli 2024 | See paper |
| MultiPICo | Casola et al. 2024 | HuggingFace |
| EPIC | Frenda et al. 2023 | HuggingFace |
| MHS (Measuring Hate Speech) | Sachdeva et al. 2022 | [HuggingFace](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |

The aggregated parquet files are expected under `results/sarcasm/` at the repository root. The expected schema for each is documented inside the respective run script.

---

## Directory layout

Run every script **from the repository root** (this directory). Each script resolves its data root as
`BASE = Path(__file__).parent.parent`, which points to the repository root regardless of the sub-directory the script lives in, so `results/` and `figures/` are always resolved relative to this root.

```
contest-sarcasm-uncertainty/     # repository root
├── experiments/
├── analysis/
├── plotting/
├── requirements.txt
├── .env.example
├── .env                         # your API keys (not committed)
├── results/                     # inputs/outputs (not committed)
│   ├── sarcasm/                 # aggregated corpus parquets
│   ├── llm_e1/                  # E1 outputs
│   ├── llm_e2/                  # E2 outputs
│   ├── llm_e3/                  # E3 outputs + CSVs
│   ├── llm_hate/                # MHS outputs
│   ├── metareview/              # distribution metrics, cost accounting, learned probe
│   └── strong_baselines/        # strong-baseline results
└── figures/                     # generated figures (not committed)
```

Neither `results/` nor `figures/` is redistributed; create them locally.

---

## Reproducing main results

### E1 (frontier models, sarcasm)

```bash
python experiments/run_e1.py --n 400 --phase 1 \
    --datasets CSC,MultiPICo,EPIC --providers gpt,claude \
    --gpt_model gpt-4o-mini --claude_model anthropic/claude-haiku-4.5
```

Frontier models (`--gpt_client openrouter`):

```bash
python experiments/run_e1.py --n 400 --phase frontier400 \
    --gpt_model openai/gpt-4.1 --claude_model anthropic/claude-opus-4.8 \
    --gpt_client openrouter
```

### E3 analysis (Qwen2.5 sweep — requires pre-run CSVs)

Capacity-sweep CSVs (`e3_q0p5b_CSC.csv`, ..., `e3_q32b_EPIC.csv`) are not included. The E3 GPU sweep was run on a compute cluster; place the resulting CSVs in `results/llm_e3/`.

```bash
python analysis/run_e3_analyze.py   # reads results/llm_e3/e3_q*.csv
```

### E3 cross-family (via OpenRouter API)

```bash
python experiments/run_e3_xfamily_api.py \
    --families qwen3,gemma3,llama3x \
    --corpora CSC,MultiPICo,EPIC
python analysis/run_e3_xfamily_analyze.py
```

### Method (AUROC / AUCAC) and baselines

```bash
python analysis/run_e3_method.py
python analysis/run_metareview_costaccounting.py
python analysis/run_strong_baselines.py
```

### Figures

```bash
python plotting/plot_capacity_curves.py
python plotting/plot_costacc.py
python plotting/plot_distmetrics.py
```

---

## Reproducibility notes

- All random seeds are set to `seed=42`.
- Bootstrap tests use 4000 resamples (main sweep) or 2000 resamples (AUROC CI).
- The prediction threshold for binary classification is `p_yes >= 0.5` (inclusive).
- Human-disagreement terciles use a rank-based split: `np.digitize(rankdata(dis_mi)/(n+1), [1/3, 2/3])`. This differs from quantile-based splits for corpora with discrete `dis_mi` values (e.g., EPIC).

---

## Citation

If you use this code, please cite:

```
[citation to be added upon publication]
```

---

## License

MIT License. See LICENSE file (to be added upon publication).
