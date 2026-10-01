# legacy_aaai/

Scripts from the earlier conference version of this work whose outputs are **not**
used by the KBS manuscript ("An Annotation-Grounded Differential-Contraction
Framework for Uncertainty Diagnosis in Large Language Models"). They are kept for
transparency only; the KBS numbers and figures are produced by `kbs/` and
`experiments/` (see the top-level README).

- `analysis/`   – 400-item E3 analyses, strong baselines, distribution metrics,
                  cost accounting and learned-probe experiments.
- `plotting/`   – figures of the earlier version.
- `experiments/run_hate.py` – E1/E2 on the Measuring Hate Speech corpus.

Paths resolve relative to the repository root (`BASE = Path(__file__).resolve().parents[2]`)
or to the current working directory (`results/...`); run them from the repository root.
These scripts have not been re-run for the KBS release.
