"""Central path configuration for the CONTEST code release.

Every KBS-pipeline script resolves its input and output locations through this
module instead of hard-coded absolute paths.

Layout (all relative to WORK_DIR):
    data/raw/        third-party corpora (not redistributed; see README)
    results/         pipeline inputs/outputs; the small derived result files that
                     make_numbers.py / make_figures.py read are shipped here
    paper_out/       numbers.tex and figures written by kbs/paper/*.py

Environment variables (all optional):
    CONTEST_WORK_DIR   root that contains data/ and results/   (default: repository root)
    CONTEST_PAPER_OUT  output directory for numbers.tex/figures (default: <WORK_DIR>/paper_out)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent

WORK_DIR: Path = Path(os.environ.get("CONTEST_WORK_DIR", str(REPO_ROOT))).expanduser().resolve()
DATA_DIR: Path = WORK_DIR / "data"
RESULTS_DIR: Path = WORK_DIR / "results"
PAPER_OUT: Path = Path(
    os.environ.get("CONTEST_PAPER_OUT", str(WORK_DIR / "paper_out"))
).expanduser().resolve()
