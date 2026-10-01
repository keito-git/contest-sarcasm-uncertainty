#!/usr/bin/env python3
"""Fail unless every required result macro is actually used in the manuscript text.

Guards against result macros that are defined in numbers.tex but never reported
in the manuscript text.

Usage: python kbs/paper/check_required_macros.py [en|ja]   (default: both)

The manuscript LaTeX sources are not part of this repository. Point
CONTEST_TEX_DIR to a directory laid out as <lang>/numbers.tex,
<lang>/sections/*.tex and <lang>/main_<lang>.tex (default: paper_out/).
Exit code 1 if any required macro family is missing.
"""
import glob
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contest_paths import PAPER_OUT  # noqa: E402  (repo-root path config)

ROOT = Path(os.environ.get("CONTEST_TEX_DIR", str(PAPER_OUT)))
CORPORA = ["CSC", "EPIC", "ES", "DE", "FR", "IT"]

# Each entry: (description, list of macro names that must ALL appear in the text).
REQUIRED = [
    ("C1 ΔAsym bootstrap CIs (all corpora)",
     [f"DeltaAsym{c}CIlo" for c in CORPORA] + [f"DeltaAsym{c}CIhi" for c in CORPORA]),
    ("C2a soft-label ΔAsym (Brier) with CIs",
     [f"DeltaAsymBrier{c}" for c in CORPORA]),
    ("C2a soft-label ΔAsym (JSD)",
     [f"DeltaAsymJSD{c}" for c in CORPORA]),
    ("C2b v5 reference-label-noise null (CSC strata fixed; Holm; explained share)",
     [f"NullFiveMean{c}" for c in CORPORA] + [f"NullFivePHolm{c}" for c in CORPORA]
     + [f"NullFiveBandLo{c}" for c in CORPORA] + [f"NullFiveBandHi{c}" for c in CORPORA]
     + [f"NullFiveExplainedPct{c}" for c in CORPORA]),
    ("C2c v3 per-tercile soft-label changes (over-confidence on high-dis items)",
     [f"SoftDropLowBrier{c}" for c in CORPORA] + [f"SoftDropHighBrier{c}" for c in CORPORA]),
    ("C3 real annotator counts (non-English MultiPICo)",
     [f"ki{c}Med" for c in ["ES", "DE", "FR", "IT"]]),
    ("C4 stratum instability",
     [f"InstabFrac{c}" for c in CORPORA]),
    ("C4 ΔAsym restricted to k_i >= 4",
     [f"DeltaAsymKiGeFour{c}" for c in ["ES", "DE", "FR", "IT"]]),
    ("C5 endpoint variants",
     [f"EndpointDAOneFiveToThirtyTwo{c}" for c in CORPORA]
     + [f"EndpointDAThreeToThirtyTwo{c}" for c in CORPORA]
     + [f"EndpointDAZeroFiveToFourteen{c}" for c in CORPORA]),
    ("C6 14B probe variant",
     [f"ProbeFourteenBauroc{c}" for c in CORPORA]),
    ("C7 v3 cost-accuracy (one definition; random band, confidence, oracle, probe, CIs of differences)",
     [f"AUBACvThreeRand{c}" for c in CORPORA] + [f"AUBACvThreeOracle{c}" for c in CORPORA]
     + [f"AUBACvThreeCont{c}" for c in CORPORA] + [f"AUBACvThreeConf{c}" for c in CORPORA]
     + [f"AUBACvThreeDeltaCont{c}CIlo" for c in CORPORA] + [f"AUBACvThreeDeltaCont{c}CIhi" for c in CORPORA]),
    ("CSC floor sensitivity",
     ["FloorDropBothDeltaAsymCSC", "FloorDropBothDeltaAsymCSCCIlo", "FloorDropBothDeltaAsymCSCCIhi",
      "FloorNDenomCSC", "FloorNClipCSC"]),
    ("C9 comparison methods",
     [f"GroupingLossAUROC{c}" for c in CORPORA] + [f"CalibConfAUROC{c}" for c in CORPORA]
     + [f"DivVarAUROC{c}" for c in CORPORA] + [f"QwenVarAUROC{c}" for c in CORPORA]),
    ("Single-model entropy baseline for escalation",
     [f"BaseHAUROC{c}" for c in CORPORA]),
    ("Bootstrap counts actually used",
     ["nBootDeltaAsym"]),
]


def active_text(lang: str) -> str:
    base = ROOT / lang
    files = sorted(glob.glob(str(base / "sections" / "*.tex")))
    files = [f for f in files if not re.search(r"_v\d|bak|backup", Path(f).name)]
    main = base / ("main_en.tex" if lang == "en" else "main_ja.tex")
    files.append(str(main))
    return "".join(Path(f).read_text(encoding="utf-8") for f in files)


def defined(lang: str) -> set:
    t = (ROOT / lang / "numbers.tex").read_text(encoding="utf-8")
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", t))


def check(lang: str) -> bool:
    text = active_text(lang)
    defs = defined(lang)
    ok = True
    print(f"== {lang} ==")
    for desc, macros in REQUIRED:
        missing_def = [m for m in macros if m not in defs]
        unused = [m for m in macros if m in defs and not re.search(r"\\" + m + r"(?![A-Za-z])", text)]
        status = "OK" if not missing_def and not unused else "MISSING"
        if status != "OK":
            ok = False
        print(f"  [{status}] {desc}")
        if missing_def:
            print(f"      not defined in numbers.tex: {missing_def}")
        if unused:
            print(f"      defined but not used in text: {unused}")
    return ok


if __name__ == "__main__":
    langs = sys.argv[1:] or ["en", "ja"]
    results = [check(l) for l in langs]
    sys.exit(0 if all(results) else 1)
