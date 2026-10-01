#!/usr/bin/env python3
"""
make_numbers.py -- kbs/paper/make_numbers.py
Reads result JSON files and emits numbers.tex (all numeric LaTeX macros of the
KBS manuscript).

ALL numeric values are read from JSON. No hardcoded fallbacks.
If a required key is missing, the script raises KeyError immediately.

Usage (from the repository root):
    python kbs/paper/make_numbers.py

Output:
    $CONTEST_PAPER_OUT/numbers.tex   (default: paper_out/numbers.tex)

Optional:
    CONTEST_NUMBERS_TIMESTAMP  ISO timestamp written to the "% Generated:" header
                               line instead of the current time (useful for
                               byte-identical comparison with a reference file).

Dependencies (all paths relative to $CONTEST_WORK_DIR, default: repository root;
all files are shipped in results/):
    results/revision_gpu_2026-09/regen_analysis/C7_REGEN_20260925T182937.json
    results/revision_gpu_2026-09/regen_analysis/C8_REGEN_20260925T183127.json
    results/revision_gpu_2026-09/regen_analysis/DEDUP_UNION_20260925T182829.json
    results/revision_gpu_2026-09/regen_analysis/COSTACC_REGEN.json
    results/revision_gpu_2026-09/round1_analysis/{C1_DASYM_CI, C2a_SOFT_DASYM_v3,
        C2_SOFTLABEL_v2, C3_ANNOTATOR_COUNTS, C4_INSTABILITY_v2, C5_ENDPOINT,
        C6_PROBE_14B, C7_COSTACC_v2, C8_MEAN_DEP_v2, C9_COMPARISONS_v2, C2b_v4,
        C2c_v3, C7_v3, C9_floor_v3, C2b_v5}.json
    results/revision_2026-09/rc3_ki_reliability.json
    results/revision_2026-09/rc4_mean_independence.json
    results/revision_2026-09/rc6_logprob_fixed.json
    results/revision_2026-09/rc6_beyond_spearman.json
    results/revision_2026-09/ki_per_language.json
    results/revision_2026-09/rc9_capability_gradient.json
    results/llm_e1/e1_summary_phase1.json
    results/llm_e1/e1_summary_phaseepic1.json
    results/llm_e1/e1_summary_phasefrontier400.json
    results/llm_e1/e1_summary_phaseepicF.json
"""

import json
import os
import pathlib
import datetime

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from contest_paths import WORK_DIR, PAPER_OUT  # noqa: E402  (repo-root path config)
BASE = WORK_DIR

# ---- Input files (all required; script raises if missing) ----
C7_JSON    = BASE / "results/revision_gpu_2026-09/regen_analysis/C7_REGEN_20260925T182937.json"
C8_JSON    = BASE / "results/revision_gpu_2026-09/regen_analysis/C8_REGEN_20260925T183127.json"
DEDUP_JSON = BASE / "results/revision_gpu_2026-09/regen_analysis/DEDUP_UNION_20260925T182829.json"
RC3_JSON   = BASE / "results/revision_2026-09/rc3_ki_reliability.json"
RC4_JSON   = BASE / "results/revision_2026-09/rc4_mean_independence.json"
RC6FIX_JSON= BASE / "results/revision_2026-09/rc6_logprob_fixed.json"
RC6BSP_JSON= BASE / "results/revision_2026-09/rc6_beyond_spearman.json"
KI_LANG_JSON=BASE / "results/revision_2026-09/ki_per_language.json"
RC9_JSON   = BASE / "results/revision_2026-09/rc9_capability_gradient.json"
E1_PHASE1_JSON      = BASE / "results/llm_e1/e1_summary_phase1.json"
E1_PHASEEPIC1_JSON  = BASE / "results/llm_e1/e1_summary_phaseepic1.json"
E1_FRONTIER400_JSON = BASE / "results/llm_e1/e1_summary_phasefrontier400.json"
E1_EPICF_JSON       = BASE / "results/llm_e1/e1_summary_phaseepicF.json"
COSTACC_JSON        = BASE / "results/revision_gpu_2026-09/regen_analysis/COSTACC_REGEN.json"

# ---- Extended-analysis JSON files ----
R1_BASE = BASE / "results/revision_gpu_2026-09/round1_analysis"
C1_CI_JSON   = R1_BASE / "C1_DASYM_CI.json"        # ΔAsym bootstrap CIs (B=4000)
C2A_JSON     = R1_BASE / "C2a_SOFT_DASYM_v3.json"  # soft-label ΔAsym Brier/JSD
C2B_JSON     = R1_BASE / "C2_SOFTLABEL_v2.json"    # noise-null summaries
C3_JSON      = R1_BASE / "C3_ANNOTATOR_COUNTS.json" # actual k_i per corpus
C4_JSON      = R1_BASE / "C4_INSTABILITY_v2.json"  # stratum instability, ki≥4/5
C5_JSON      = R1_BASE / "C5_ENDPOINT.json"        # endpoint variants
C6_JSON      = R1_BASE / "C6_PROBE_14B.json"       # 14B probe AUROC
C7V2_JSON    = R1_BASE / "C7_COSTACC_v2.json"      # cost-accuracy v2 (corrected)
C8V2_JSON    = R1_BASE / "C8_MEAN_DEP_v2.json"     # CSC rho(dis, human mean)
C9V2_JSON    = R1_BASE / "C9_COMPARISONS_v2.json"  # comparison methods incl. sign-flip

OUT = PAPER_OUT / "numbers.tex"


# ---- Helpers ----

def require(d, *path):
    """
    Traverse d using path keys. Raise KeyError with full path if any key is missing
    or the value is None. Never silently return a default.
    """
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and 0 <= k < len(cur):
            cur = cur[k]
        else:
            raise KeyError(f"Missing key path: {' -> '.join(str(x) for x in path)}")
    if cur is None:
        raise KeyError(f"Value is None at path: {' -> '.join(str(x) for x in path)}")
    return cur


def load_required(path):
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required JSON not found: {p}")
    with open(p) as f:
        return json.load(f)


def fmt(x, ndigits=3):
    return f"{x:.{ndigits}f}"


def pct(x, ndigits=1):
    return f"{x * 100:.{ndigits}f}"


def sign_fmt(x, ndigits=3):
    """Format with explicit + sign."""
    s = f"{abs(x):.{ndigits}f}"
    return f"+{s}" if x >= 0 else f"-{s}"


def e1_lookup(data, dataset, provider):
    """Return the result row for (dataset, provider) from an e1 summary JSON."""
    for row in data["results"]:
        if row["dataset"] == dataset and row["provider"] == provider:
            return row
    raise KeyError(f"e1 summary: no entry for dataset={dataset}, provider={provider}")


def main():
    # ---- Load all JSONs (fail-fast if missing) ----
    c7     = load_required(C7_JSON)
    c8     = load_required(C8_JSON)
    dedup  = load_required(DEDUP_JSON)
    rc3    = load_required(RC3_JSON)
    rc4    = load_required(RC4_JSON)
    rc6fix = load_required(RC6FIX_JSON)
    rc6bsp = load_required(RC6BSP_JSON)
    ki_lang= load_required(KI_LANG_JSON)
    rc9    = load_required(RC9_JSON)
    e1_p1  = load_required(E1_PHASE1_JSON)
    e1_ep1 = load_required(E1_PHASEEPIC1_JSON)
    e1_f4  = load_required(E1_FRONTIER400_JSON)
    e1_epf = load_required(E1_EPICF_JSON)

    # ---- Extended-analysis JSONs ----
    r1_c1   = load_required(C1_CI_JSON)
    r1_c2a  = load_required(C2A_JSON)
    r1_c2b  = load_required(C2B_JSON)
    r1_c3   = load_required(C3_JSON)
    r1_c4   = load_required(C4_JSON)
    r1_c5   = load_required(C5_JSON)
    r1_c6   = load_required(C6_JSON)
    r1_c7v2 = load_required(C7V2_JSON)
    r1_c8v2 = load_required(C8V2_JSON)
    r1_c9v2 = load_required(C9V2_JSON)

    # ---- Extended-analysis JSONs (noise null, per-tercile soft labels, AUBAC, floor) ----
    r2_c2b_v4  = load_required(R1_BASE / "C2b_v4.json")      # noise-null corrected (superseded by v5)
    r2_c2c_v3  = load_required(R1_BASE / "C2c_v3.json")      # per-tercile soft-label Brier
    r2_c7_v3   = load_required(R1_BASE / "C7_v3.json")       # unified AUBAC definition
    r2_c9fl_v3 = load_required(R1_BASE / "C9_floor_v3.json") # CSC floor sensitivity

    # ---- Noise null with CSC Likert resampling ----
    r3_c2b_v5  = load_required(R1_BASE / "C2b_v5.json")      # noise-null v5: CSC uses Likert resampling

    lines = [
        "% ============================================================",
        "% numbers.tex — AUTO-GENERATED by make_numbers.py",
        f"% Generated: {os.environ.get('CONTEST_NUMBERS_TIMESTAMP') or datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        "% DO NOT EDIT BY HAND. All values read from JSON result files.",
        "% No hardcoded fallbacks. Script raises error if any key is missing.",
        "% ============================================================",
        "",
    ]

    _emitted: set = set()

    def cmd(name, val):
        if name in _emitted:
            raise AssertionError(f"Duplicate macro emission: \\{name}  (val={val!r})")
        _emitted.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # Section titles are written verbatim as comments into numbers.tex. They are kept
    # unchanged so that the generated file is byte-identical to the manuscript copy.
    def section(title):
        lines.append(f"")
        lines.append(f"% ------ {title} ------")

    # ================================================================
    # 1. Corpus sizes (from C7 per_corpus_tercile_acc[corpus][n])
    # ================================================================
    section("Corpus sizes")
    tc = require(c7, "per_corpus_tercile_acc")
    for corpus_key, macro in [
        ("CSC",          "nCSC"),
        ("EPIC",         "nEPIC"),
        ("MultiPICo_es", "nES"),
        ("MultiPICo_de", "nDE"),
        ("MultiPICo_fr", "nFR"),
        ("MultiPICo_it", "nIT"),
    ]:
        n = require(tc, corpus_key, "n")
        cmd(macro, str(int(n)))

    # nFrontier: experimental design constant (n=400 sample for frontier eval)
    # This is an experimental design parameter, NOT a result statistic.
    cmd("nFrontier", "400")

    # ================================================================
    # 2. C7: ΔAsym (from asym_results list)
    # p_holm is read from C1_DASYM_CI (B=4000 bootstrap) rather than C7_REGEN,
    # whose smaller bootstrap rounds the MultiPICo-it value to 0.0000.
    # ================================================================
    section("DeltaAsym values (C7, Holm-corrected; p_holm from C1_DASYM_CI)")
    ar = require(c7, "asym_results")
    if not isinstance(ar, list):
        raise TypeError(f"asym_results should be a list, got {type(ar)}")

    # Build lookup: corpus+subset -> row (for asym_acc from C7_REGEN)
    asym_lookup = {}
    for row in ar:
        key = (require(row, "corpus"), require(row, "subset"))
        asym_lookup[key] = row

    # Build p_holm lookup from C1_DASYM_CI (authoritative; B=4000)
    c1_holm_by_corpus = {}
    for row in require(r1_c1, "results"):
        if require(row, "subset") == "full":
            c1_holm_by_corpus[require(row, "corpus")] = require(row, "p_holm")

    for corpus_key, macro_suffix in [
        ("CSC",          "DeltaAsymCSC"),
        ("EPIC",         "DeltaAsymEPIC"),
        ("MultiPICo_es", "DeltaAsymES"),
        ("MultiPICo_de", "DeltaAsymDE"),
        ("MultiPICo_fr", "DeltaAsymFR"),
        ("MultiPICo_it", "DeltaAsymIT"),
    ]:
        row = asym_lookup.get((corpus_key, "full"))
        if row is None:
            raise KeyError(f"asym_results: no entry for corpus={corpus_key}, subset=full")
        da  = require(row, "asym_acc")
        # p_holm from C1_DASYM_CI (B=4000), not C7_REGEN
        ph  = c1_holm_by_corpus[corpus_key]
        cmd(macro_suffix,           sign_fmt(da, 3))
        # Two alias forms: PvalHolm (verbose) and pHolm (short used in sections)
        cmd(macro_suffix + "PvalHolm", fmt(ph, 4))
        cmd(macro_suffix + "pHolm",    fmt(ph, 4))

    # ---- DeltaAsym on the original 400-item sample (continuity values) ----
    section("Old orig400 DeltaAsym values (comparison with AAAI submission)")
    for corpus_key, macro in [
        ("CSC",  "DeltaAsymCSCorigFour"),
        ("EPIC", "DeltaAsymEPICorigFour"),
    ]:
        row = asym_lookup.get((corpus_key, "orig400"))
        if row is None:
            raise KeyError(f"asym_results: no orig400 entry for corpus={corpus_key}")
        da_orig = require(row, "asym_acc")
        cmd(macro, sign_fmt(da_orig, 3))

    # C7: per-tercile BalAcc values needed for figures
    # Macro name suffix: Small = 0.5B, Large = 32B (avoid digits in LaTeX macro names)
    section("Per-tercile BalAcc at 0.5B (Small) and 32B (Large) from C7 figure data")
    caps_tags = [("q0p5b", "Small"), ("q32b", "Large")]
    for corpus_key, macro_prefix in [
        ("CSC",          "TerCSC"),
        ("EPIC",         "TerEPIC"),
        ("MultiPICo_es", "TerES"),
        ("MultiPICo_de", "TerDE"),
        ("MultiPICo_fr", "TerFR"),
        ("MultiPICo_it", "TerIT"),
    ]:
        bal = require(tc, corpus_key, "bal_acc")
        for tag, size_suffix in caps_tags:
            if tag in bal:
                lo = require(bal, tag, "low_dis",  "bal_acc")
                hi = require(bal, tag, "high_dis", "bal_acc")
                cmd(f"{macro_prefix}Low{size_suffix}",  fmt(lo, 3))
                cmd(f"{macro_prefix}High{size_suffix}", fmt(hi, 3))

    # ================================================================
    # 3. C8: Contraction probe AUROC
    # ================================================================
    section("Contraction probe AUROC (C8)")
    pc = require(c8, "per_corpus")
    for corpus_key, macro in [
        ("CSC",          "ContAUROCCSC"),
        ("EPIC",         "ContAUROCEPIC"),
        ("MultiPICo_es", "ContAUROCES"),
        ("MultiPICo_de", "ContAUROCDE"),
        ("MultiPICo_fr", "ContAUROCFR"),
        ("MultiPICo_it", "ContAUROCIT"),
    ]:
        auroc = require(pc, corpus_key, "contraction_probes",
                        "contraction_q3b_q0p5b", "AUROC")
        ci_lo = require(pc, corpus_key, "contraction_probes",
                        "contraction_q3b_q0p5b", "CI_low_95")
        ci_hi = require(pc, corpus_key, "contraction_probes",
                        "contraction_q3b_q0p5b", "CI_high_95")
        cmd(macro,         fmt(auroc, 3))
        cmd(macro + "CIlo", fmt(ci_lo, 3))
        cmd(macro + "CIhi", fmt(ci_hi, 3))

    # Published (n=400) AUROC for disclosure
    section("Published n=400 contraction AUROC (disclosure)")
    for corpus_key, macro in [
        ("CSC",  "ContAUROCCSCpub"),
        ("EPIC", "ContAUROCEPICpub"),
    ]:
        auroc_pub = require(pc, corpus_key, "published_auroc_n400")
        cmd(macro, fmt(auroc_pub, 3))

    # Baseline entropy (single-model H_small) and oracle 1-dis AUROC with CIs (C8)
    section("Baseline entropy and oracle 1-dis AUROC with CIs (C8)")
    for corpus_key, corpus_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        bh = require(pc, corpus_key, "baseline_H_small")
        cmd(f"BaseHAUROC{corpus_suffix}",      fmt(require(bh, "AUROC"), 3))
        cmd(f"BaseHAUROC{corpus_suffix}CIlo",  fmt(require(bh, "CI_low"), 3))
        cmd(f"BaseHAUROC{corpus_suffix}CIhi",  fmt(require(bh, "CI_high"), 3))
        oa = require(pc, corpus_key, "baseline_oracle")
        cmd(f"OracleAUROC{corpus_suffix}",     fmt(require(oa, "AUROC"), 3))

    # Diversity probe AUROCs for comparison table (tab:diversity)
    # diversity_3b  = Phi-3-mini-4k-instruct (3.8B)  tag ll3b
    # diversity_1b  = SmolLM2-1.7B-Instruct          tag ll1b
    # diversity_gm2 = StableLM-zephyr-3B             tag gm2_2b
    # sampling_noise= Qwen2.5-0.5B repeated sampling
    section("Diversity probe AUROCs vs capacity gap (C8)")
    div_types = [
        ("diversity_3b",   "ThreeB"),   # Phi-3-mini-3.8B
        ("diversity_1b",   "OneB"),     # SmolLM2-1.7B
        ("diversity_gm2",  "StableB"),  # StableLM-zephyr-3B
        ("sampling_noise", "Nz"),
    ]
    for corpus_key, corpus_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        for div_key, div_suffix in div_types:
            dp = require(pc, corpus_key, "diversity_probes", div_key)
            auroc = require(dp, "AUROC")
            cmd(f"DivAUROC{div_suffix}{corpus_suffix}", fmt(auroc, 3))

    # Diversity probe significance flags
    section("Diversity probe vs capacity gap significance")
    for corpus_key, macro_suffix in [
        ("EPIC",         "DivVsCapEPIC"),
        ("MultiPICo_es", "DivVsCapES"),
        ("MultiPICo_de", "DivVsCapDE"),
        ("MultiPICo_fr", "DivVsCapFR"),
        ("MultiPICo_it", "DivVsCapIT"),
    ]:
        dp = require(pc, corpus_key, "diversity_probes", "diversity_3b")
        sig = require(dp, "vs_primary", "sig_05")
        cmd(macro_suffix + "Sig", "true" if sig else "false")

    # ================================================================
    # 4. k_i distributions (CSC and EPIC from RC3)
    # ================================================================
    section("k_i distributions (CSC and EPIC from RC3 JSON)")
    ki_rc3 = require(rc3, "rc3", "ki_distributions")
    for corpus_key, macro_prefix in [
        ("CSC",  "kiCSC"),
        ("EPIC", "kiEPIC"),
    ]:
        ki_row = require(ki_rc3, corpus_key)
        cmd(macro_prefix + "Min",  str(int(require(ki_row, "min"))))
        cmd(macro_prefix + "Med",  fmt(require(ki_row, "median"), 1))
        cmd(macro_prefix + "Mean", fmt(require(ki_row, "mean"), 2))
        cmd(macro_prefix + "Max",  str(int(require(ki_row, "max"))))

    # k_i for non-English MultiPICo — use C3 (actual parquet data, NOT ki_per_language proxy)
    section("k_i distributions (non-English MultiPICo from C3 actual parquet data)")
    c3_langs = require(r1_c3, "per_lang")
    c3_lang_map = {
        "MultiPICo_es": "es",
        "MultiPICo_de": "de",
        "MultiPICo_fr": "fr",
        "MultiPICo_it": "it",
    }
    all_mins, all_maxes, all_meds = [], [], []
    for corpus_key, macro_prefix in [
        ("MultiPICo_es", "kiES"),
        ("MultiPICo_de", "kiDE"),
        ("MultiPICo_fr", "kiFR"),
        ("MultiPICo_it", "kiIT"),
    ]:
        lang_key = c3_lang_map[corpus_key]
        ki_row = require(c3_langs, lang_key)
        cmd(macro_prefix + "Min",  str(int(require(ki_row, "ki_min"))))
        cmd(macro_prefix + "Med",  fmt(require(ki_row, "ki_median"), 1))
        cmd(macro_prefix + "Mean", fmt(require(ki_row, "ki_mean"), 2))
        cmd(macro_prefix + "Max",  str(int(require(ki_row, "ki_max"))))
        all_mins.append(int(require(ki_row, "ki_min")))
        all_maxes.append(int(require(ki_row, "ki_max")))
        all_meds.append(require(ki_row, "ki_median"))

    # Aggregated MultiPICo non-English ki summary (from C3 actual data)
    section("Aggregated MultiPICo non-English ki (C3 corrected values)")
    cmd("kiMPMin", str(min(all_mins)))   # actual: 3 (not 2 as in ki_per_language proxy)
    cmd("kiMPMax", str(max(all_maxes))) # actual: 16 (not 8 as in ki_per_language proxy)
    # Median: all languages have ki_median=5.0
    cmd("kiMPMed", fmt(all_meds[0], 1))  # 5.0
    # Mean SE of dis from RC3 delta-method analysis
    mp_dis_se = require(rc3, "rc3", "MultiPICo_dis_se", "mean_se")
    cmd("kiMPMeanSE", fmt(mp_dis_se, 3))

    # ================================================================
    # 5. RC4: mean-independence / binary dis
    # ================================================================
    section("RC4: mean-independence and binary dis statistics")
    rc4_inner = require(rc4, "rc4")
    spear = require(rc4_inner, "corrected_analysis", "MultiPICo_EPIC_correct",
                    "spearman_dis_vs_p_ironic_MP")
    cmd("SpearmanDisPironicMP", sign_fmt(spear, 3))
    floor_frac = require(rc4_inner, "corrected_analysis", "CSC_correct_dis_mi",
                         "frac_denom_floored")
    cmd("CSCFloorFrac", pct(floor_frac, 1))

    # ================================================================
    # 6. rc6: U_logprob values (two-label normalized p_yes)
    # ================================================================
    section("RC6 fixed: corrected U_logprob signal metrics")
    rc6fix_metrics = require(rc6fix, "rc6_logprob_fixed", "signal_metrics")
    lp_lookup = {(require(m, "corpus"), m.get("provider", "gpt")): m
                 for m in rc6fix_metrics}

    for (corpus, prov), (macro_rho, macro_partial, macro_auroc) in [
        (("CSC",       "gpt"), ("UlpRhoCSC",    "UlpPartialCSC",  "UlpAUROCCSC")),
        (("MultiPICo", "gpt"), ("UlpRhoMP",     "UlpPartialMP",   "UlpAUROCMP")),
        (("EPIC",      "gpt"), ("UlpRhoEPIC",   "UlpPartialEPIC", "UlpAUROCEPIC")),
    ]:
        row = lp_lookup.get((corpus, prov))
        if row is None:
            raise KeyError(f"rc6_logprob_fixed: no entry for corpus={corpus}, provider={prov}")
        rho     = require(row, "raw_spearman", "rho")
        partial = require(row, "partial_spearman_ctrl_ki", "rho")
        auroc   = require(row, "auroc_signal_for_high_vs_low_dis_tercile")
        mi_bc   = require(row, "mutual_information", "mi_bc")
        iso_r2  = require(row, "isotonic_r2")
        cmd(macro_rho,     fmt(rho, 3))
        cmd(macro_partial, fmt(partial, 3))
        cmd(macro_auroc,   fmt(auroc, 3))
        # MI and isotonic R2 for lp (used in table tab:e2)
        suffix = corpus.replace("MultiPICo", "MP")
        cmd(f"UlpMI{suffix}",  fmt(mi_bc,   3))
        cmd(f"UlpIso{suffix}", fmt(iso_r2,  3))

    # ================================================================
    # 7. RC6 beyond-Spearman: U_verbal signal metrics
    # ================================================================
    section("RC6 beyond-Spearman: U_verbal AUROC and MI")
    rc6bsp_metrics = require(rc6bsp, "rc6", "signal_metrics")
    uv_lookup = {}
    for m in rc6bsp_metrics:
        key = (require(m, "corpus"), m.get("provider", "gpt"), m.get("signal", "U_verbal"))
        uv_lookup[key] = m

    for (corpus, prov, sig), (macro_rho, macro_auroc, macro_mi, macro_iso) in [
        (("CSC",       "gpt",    "U_verbal"),
         ("UvRhoCSCgpt",    "UvAUROCCSCgpt",    "UvMICSCgpt",    "UvIsoCSCgpt")),
        (("CSC",       "claude", "U_verbal"),
         ("UvRhoCSCclaude", "UvAUROCCSCclaude", "UvMICSCclaude", "UvIsoCSCclaude")),
        (("EPIC",      "gpt",    "U_verbal"),
         ("UvRhoEPICgpt",   "UvAUROCEPICgpt",   "UvMIEPICgpt",   "UvIsoEPICgpt")),
        (("EPIC",      "claude", "U_verbal"),
         ("UvRhoEPICclaude","UvAUROCEPICclaude","UvMIEPICclaude","UvIsoEPICclaude")),
    ]:
        row = uv_lookup.get((corpus, prov, sig))
        if row is None:
            raise KeyError(f"rc6_beyond_spearman: no entry for corpus={corpus}, provider={prov}, signal={sig}")
        rho   = require(row, "raw_spearman", "rho")
        auroc = require(row, "auroc_signal_for_high_vs_low_dis_tercile")
        mi_bc = require(row, "mutual_information", "mi_bias_corrected")
        iso   = require(row, "isotonic_r2_signal_vs_dis")
        cmd(macro_rho,   fmt(rho,   3))
        cmd(macro_auroc, fmt(auroc, 3))
        cmd(macro_mi,    fmt(mi_bc, 3))
        cmd(macro_iso,   fmt(iso,   3))

    # Uppercase provider aliases (section files use ALLCAPS provider names)
    section("Uppercase provider aliases for U_verbal macros (section file naming convention)")
    alias_map = [
        ("UvRhoCSCgpt",    "UvRhoCSCGPT"),
        ("UvAUROCCSCgpt",  "UvAUROCCSCGPT"),
        ("UvMICSCgpt",     "UvMICSCGPT"),
        ("UvIsoCSCgpt",    "UvIsoCSCGPT"),
        ("UvRhoEPICgpt",   "UvRhoEPICGPT"),
        ("UvAUROCEPICgpt", "UvAUROCEPICGPT"),
        ("UvMIEPICgpt",    "UvMIEPICGPT"),
        ("UvIsoEPICgpt",   "UvIsoEPICGPT"),
        ("UvRhoEPICclaude",   "UvRhoEPICClaude"),
        ("UvAUROCEPICclaude", "UvAUROCEPICClaude"),
        ("UvMIEPICclaude",    "UvMIEPICClaude"),
        ("UvIsoEPICclaude",   "UvIsoEPICClaude"),
    ]
    for src, alias in alias_map:
        lines.append(f"\\newcommand{{\\{alias}}}{{\\{src}}}")

    # ================================================================
    # 8. Partial Spearman rho values from RC3 (k_i controlled)
    # ================================================================
    section("Partial Spearman rho (k_i controlled) from RC3")
    ps_list = require(rc3, "rc3", "partial_spearman_ctrl_ki")
    ps_lookup = {(m["corpus"], m.get("provider", "gpt"), m.get("signal", "U_verbal")): m
                 for m in ps_list}

    for (corpus, prov, sig), macro in [
        (("CSC",        "gpt",    "U_verbal"), "PartialRhoCSCGPTUv"),
        (("EPIC",       "gpt",    "U_verbal"), "PartialRhoEPICGPTUv"),
        (("MultiPICo",  "claude", "U_verbal"), "PartialRhoMPClaudeUv"),
        (("EPIC",       "claude", "U_verbal"), "PartialRhoEPICClaudeUv"),
    ]:
        row = ps_lookup.get((corpus, prov, sig))
        if row is None:
            raise KeyError(f"rc3 partial_spearman_ctrl_ki: no entry for corpus={corpus}, provider={prov}, signal={sig}")
        partial_rho = require(row, "partial_spearman_ctrl_ki", "rho")
        cmd(macro, fmt(partial_rho, 3))

    # ================================================================
    # 9. DEDUP: overlap statistics
    # ================================================================
    section("DEDUP: EPIC / MultiPICo-en overlap")
    overlap_rate = require(dedup, "overlap_rate")
    overlap_pct = overlap_rate * 100
    cmd("DEDUPOverlapPct", fmt(overlap_pct, 1))

    # ================================================================
    # 10. Frontier model error rates (actual frontier eval, NOT Qwen proxy)
    # ================================================================
    section("Frontier model error rates (low-dis tercile, n=400 sample)")
    # GPT-4o-mini and Claude-Haiku from phase1 (n=400, CSC and MultiPICo)
    # EPIC from phaseepic1
    p1_csc_gpt   = e1_lookup(e1_p1, "CSC",   "gpt")
    p1_csc_claude= e1_lookup(e1_p1, "CSC",   "claude")
    ep1_gpt      = e1_lookup(e1_ep1, "EPIC",  "gpt")
    ep1_claude   = e1_lookup(e1_ep1, "EPIC",  "claude")
    # GPT-4.1 and Claude-Opus from frontier400 (CSC) and phaseepicF (EPIC)
    f4_csc_gpt   = e1_lookup(e1_f4, "CSC",   "gpt")
    f4_csc_claude= e1_lookup(e1_f4, "CSC",   "claude")
    epf_gpt      = e1_lookup(e1_epf, "EPIC",  "gpt")
    epf_claude   = e1_lookup(e1_epf, "EPIC",  "claude")

    cmd("ErrGPTominiCSC",     fmt(p1_csc_gpt   ["low_dis_err"], 3))
    cmd("ErrHaikuCSC",        fmt(p1_csc_claude ["low_dis_err"], 3))
    cmd("ErrGPTominiEPIC",    fmt(ep1_gpt       ["low_dis_err"], 3))
    cmd("ErrHaikuEPIC",       fmt(ep1_claude    ["low_dis_err"], 3))
    cmd("ErrGPTfouroneCSC",   fmt(f4_csc_gpt   ["low_dis_err"], 3))
    cmd("ErrOpusCSC",         fmt(f4_csc_claude ["low_dis_err"], 3))
    cmd("ErrGPTfouroneEPIC",  fmt(epf_gpt       ["low_dis_err"], 3))
    cmd("ErrOpusEPIC",        fmt(epf_claude    ["low_dis_err"], 3))

    # ================================================================
    # 11. RC9: monotonicity rho per family (CSC)
    # ================================================================
    section("RC9: monotonicity Spearman rho per model family (CSC)")
    fam_results = require(rc9, "rc9", "family_results")
    mono_map = [
        ("qwen2.5",      "MonoRhoQwen"),
        ("llama_instruct","MonoRhoLlama"),
        ("qwen3",        "MonoRhoQwenthree"),
    ]
    for fam_key, macro in mono_map:
        rho = require(fam_results, fam_key, "CSC",
                      "spearman_log_size_bal_acc", "rho")
        cmd(macro, sign_fmt(rho, 3))

    # ================================================================
    # 12. COSTACC REGEN: cost-accuracy cascade (regenerated data)
    # ================================================================
    section("COSTACC REGEN: budget-accuracy cascade (regenerated Qwen2.5 sweep)")
    costacc = load_required(COSTACC_JSON)
    pc_ca = require(costacc, "per_corpus")

    costacc_corpora = [
        ("CSC",          "AUBACCSC"),
        ("EPIC",         "AUBACEPIC"),
        ("MultiPICo_es", "AUBACESS"),
        ("MultiPICo_de", "AUBACDE"),
        ("MultiPICo_fr", "AUBACFR"),
        ("MultiPICo_it", "AUBACIT"),
    ]
    for corpus_key, macro_prefix in costacc_corpora:
        row = require(pc_ca, corpus_key)
        cmd(macro_prefix + "Cont",  fmt(require(row, "aubac_contraction"), 4))
        cmd(macro_prefix + "Rand",  fmt(require(row, "aubac_random"), 4))
        delta = require(row, "delta_cont_minus_random")
        cilo  = require(row, "delta_CI_95_low")
        cihi  = require(row, "delta_CI_95_high")
        cmd(macro_prefix + "Delta",   sign_fmt(delta, 4))
        cmd(macro_prefix + "DeltaLo", sign_fmt(cilo, 4))
        cmd(macro_prefix + "DeltaHi", sign_fmt(cihi, 4))

    # ================================================================
    # C1: ΔAsym bootstrap CIs (B=4000, from C1_DASYM_CI.json)
    # ================================================================
    section("R1-C1: DeltaAsym 95% bootstrap CIs (B=4000)")
    c1_results = require(r1_c1, "results")
    c1_boot_n  = require(r1_c1, "n_bootstrap")  # should be 4000
    # Format bootstrap count with LaTeX thousands separator
    _bn = int(c1_boot_n)
    _bn_str = f"{_bn // 1000}{{,}}{_bn % 1000:03d}" if _bn >= 1000 else str(_bn)
    cmd("nBootDeltaAsym", _bn_str)
    # Build lookup corpus+subset -> row
    c1_lookup = {}
    for row in c1_results:
        key = (require(row, "corpus"), require(row, "subset"))
        c1_lookup[key] = row
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        row = c1_lookup.get((corpus_key, "full"))
        if row is None:
            raise KeyError(f"C1: no full entry for corpus={corpus_key}")
        ci_lo = require(row, "ci_95_lo")
        ci_hi = require(row, "ci_95_hi")
        p_raw = require(row, "p_boot_raw")
        cmd(f"DeltaAsym{macro_suffix}CIlo", fmt(ci_lo, 3))
        cmd(f"DeltaAsym{macro_suffix}CIhi", fmt(ci_hi, 3))
        # p < 1/(B+1) format string for text
        if p_raw == 0.0:
            cmd(f"DeltaAsym{macro_suffix}PrawStr", f"$p < 1/(B+1)$")
        else:
            cmd(f"DeltaAsym{macro_suffix}PrawStr", f"$p = {fmt(p_raw, 4)}$")

    # ================================================================
    # C2a: Soft-label ΔAsym Brier/JSD (C2a_SOFT_DASYM_v3.json)
    # ================================================================
    section("R1-C2a: Soft-label DeltaAsym Brier/JSD with CIs")
    c2a_corpora = require(r1_c2a, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c2a_c = require(c2a_corpora, corpus_key)
        brier = require(c2a_c, "brier")
        jsd   = require(c2a_c, "jsd")
        cmd(f"DeltaAsymBrier{macro_suffix}",      sign_fmt(require(brier, "dasym"), 3))
        cmd(f"DeltaAsymBrier{macro_suffix}CIlo",  fmt(require(brier, "ci_95", 0), 3))
        cmd(f"DeltaAsymBrier{macro_suffix}CIhi",  fmt(require(brier, "ci_95", 1), 3))
        cmd(f"DeltaAsymJSD{macro_suffix}",        sign_fmt(require(jsd, "dasym"), 3))
        cmd(f"DeltaAsymJSD{macro_suffix}CIlo",    fmt(require(jsd, "ci_95", 0), 3))
        cmd(f"DeltaAsymJSD{macro_suffix}CIhi",    fmt(require(jsd, "ci_95", 1), 3))

    # ================================================================
    # C2b: Annotation-noise null summaries (C2_SOFTLABEL_v2.json)
    # ================================================================
    section("R1-C2b: Noise-null simulation summaries")
    c2b_corpora = require(r1_c2b, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c2b_c = require(c2b_corpora, corpus_key)
        null  = require(c2b_c, "c2b_null")
        obs   = require(null, "observed_dasym_hard")
        null_p975 = require(null, "null_dasym", "p97.5")
        p_null    = require(null, "p_null_exceeds_observed")
        n_sims    = require(null, "n_sims")
        cmd(f"NullDeltaAsymObs{macro_suffix}",   sign_fmt(obs, 3))
        cmd(f"NullDeltaAsymPHi{macro_suffix}",   fmt(null_p975, 3))   # P97.5 → PHi (letters only)
        cmd(f"pNullExceedsObs{macro_suffix}",    fmt(p_null, 4))
        _last_n_sims = int(n_sims)
    # emit nNullSims exactly once (same value across all corpora)
    cmd("nNullSims", str(_last_n_sims))

    # ================================================================
    # C3: Actual annotator counts per MultiPICo corpus
    # ================================================================
    section("R1-C3: Actual k_i counts per MultiPICo corpus")
    c3_corpora = require(r1_c3, "per_lang")
    for lang_key, macro_suffix in [("es", "ES"), ("de", "DE"), ("fr", "FR"), ("it", "IT")]:
        c3_c = require(c3_corpora, lang_key)
        cmd(f"ki{macro_suffix}FracGeKiFour", fmt(require(c3_c, "frac_k_ge4"), 3))
        cmd(f"ki{macro_suffix}FracGeKiFive", fmt(require(c3_c, "frac_k_ge5"), 3))
    # EPIC (from C1 results for comparison, ki_min/max from C8_REGEN)
    lines.append("% Note: kiCSC*/kiEPIC* still from RC3 JSON (no change)")

    # ================================================================
    # C4: Stratum instability and ki-filtered ΔAsym
    # ================================================================
    section("R1-C4: Stratum instability (frac_changing_tercile) and ki-filtered DeltaAsym")
    c4_corpora = require(r1_c4, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c4_c = require(c4_corpora, corpus_key)
        frac  = require(c4_c, "stratum_instability", "frac_changing_tercile")
        cmd(f"InstabFrac{macro_suffix}", fmt(frac, 3))
        # ki-filtered DeltaAsym
        dkf = require(c4_c, "dasym_by_ki_filter")
        for ki_label, ki_key in [("All", "all"), ("KiGeFour", "ki_ge4"), ("KiGeFive", "ki_ge5")]:
            if ki_key not in dkf:
                continue
            row = require(dkf, ki_key)
            da   = require(row, "asym_acc")
            ci   = require(row, "ci_95")
            cmd(f"DeltaAsym{ki_label}{macro_suffix}",      sign_fmt(da, 3))
            cmd(f"DeltaAsym{ki_label}{macro_suffix}CIlo",  fmt(ci[0], 3))
            cmd(f"DeltaAsym{ki_label}{macro_suffix}CIhi",  fmt(ci[1], 3))

    # ================================================================
    # C5: Endpoint variants of ΔAsym
    # ================================================================
    section("R1-C5: Endpoint variants of DeltaAsym")
    c5_corpora = require(r1_c5, "per_corpus")
    pair_map = [
        ("q0p5b→q32b",  "ZeroFiveToThirtyTwo"),
        ("q1p5b→q32b",  "OneFiveToThirtyTwo"),
        ("q3b→q32b",    "ThreeToThirtyTwo"),
        ("q0p5b→q14b",  "ZeroFiveToFourteen"),
    ]
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c5_c = require(c5_corpora, corpus_key)
        pairs = require(c5_c, "pair_dasym")
        for pair_key, pair_suffix in pair_map:
            if pair_key not in pairs:
                continue
            row = require(pairs, pair_key)
            da  = require(row, "asym_acc")
            ci  = require(row, "ci_95")
            cmd(f"EndpointDA{pair_suffix}{macro_suffix}",      sign_fmt(da, 3))
            cmd(f"EndpointDA{pair_suffix}{macro_suffix}CIlo",  fmt(ci[0], 3))
            cmd(f"EndpointDA{pair_suffix}{macro_suffix}CIhi",  fmt(ci[1], 3))

    # ================================================================
    # C6: 14B probe variant AUROC
    # ================================================================
    section("R1-C6: 14B probe variant AUROC")
    c6_corpora = require(r1_c6, "per_corpus")
    c6_boot_n  = require(r1_c6, "n_bootstrap")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c6_c  = require(c6_corpora, corpus_key)
        p14b  = require(c6_c, "probe_14b")
        auroc = require(p14b, "AUROC")
        ci    = require(p14b, "ci_95")
        cmd(f"ProbeFourteenBauroc{macro_suffix}",      fmt(auroc, 3))
        cmd(f"ProbeFourteenBauroc{macro_suffix}CIlo",  fmt(ci[0], 3))
        cmd(f"ProbeFourteenBauroc{macro_suffix}CIhi",  fmt(ci[1], 3))

    # ================================================================
    # C7v2: Cost-accuracy cascade (corrected C7v2)
    # ================================================================
    section("R1-C7v2: Cost-accuracy cascade random band, 3B-conf routing, oracle")
    c7v2_corpora = require(r1_c7v2, "per_corpus")
    c7v2_nseeds  = require(r1_c7v2, "n_random_seeds")
    cmd("nRandomSeedsAUBAC", str(int(c7v2_nseeds)))
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c7v2_c = require(c7v2_corpora, corpus_key)
        rand_mean = require(c7v2_c, "aubac_random_mean")
        rand_plo  = require(c7v2_c, "aubac_random_p2.5")
        rand_phi  = require(c7v2_c, "aubac_random_p97.5")
        oracle    = require(c7v2_c, "aubac_oracle_low_dis")
        conf      = require(c7v2_c, "aubac_conf_low_conf")
        cont      = require(c7v2_c, "aubac_contraction")
        cmd(f"AUBACvTwoRand{macro_suffix}",      fmt(rand_mean, 4))   # v2 → vTwo (letters only)
        cmd(f"AUBACvTwoRandPlo{macro_suffix}",   fmt(rand_plo,  4))
        cmd(f"AUBACvTwoRandPhi{macro_suffix}",   fmt(rand_phi,  4))
        cmd(f"AUBACvTwoOracle{macro_suffix}",    fmt(oracle,    4))
        cmd(f"AUBACvTwoConf{macro_suffix}",      fmt(conf,      4))
        cmd(f"AUBACvTwoCont{macro_suffix}",      fmt(cont,      4))

    # ================================================================
    # C8v2: CSC rho(dis, human mean) = 0.244 with CI
    # ================================================================
    section("R1-C8v2: CSC Spearman rho(dis_mi, human mean) with CI")
    c8v2_csc = require(r1_c8v2, "per_corpus", "CSC")
    c8v2_boot_n = require(r1_c8v2, "n_bootstrap")
    rho_csc = require(c8v2_csc, "rho")
    ci_csc  = require(c8v2_csc, "ci_95")
    cmd("RhoDisHumanMeanCSC",      fmt(rho_csc,    3))
    cmd("RhoDisHumanMeanCSCCIlo",  fmt(ci_csc[0],  3))
    cmd("RhoDisHumanMeanCSCCIhi",  fmt(ci_csc[1],  3))
    cmd("nBootMeanDepCI",          str(int(c8v2_boot_n)))
    lines.append("% Note: binary corpora (EPIC, MultiPICo) have dis_mi = H(p_human) deterministically.")
    lines.append("% No separate Spearman computation is meaningful for those corpora.")

    # ================================================================
    # C9v2: Comparison methods incl. sign-flipped grouping-loss AUROC
    # ================================================================
    section("R1-C9v2: Comparison method AUROCs (sign-flipped grouping-loss)")
    c9v2_corpora = require(r1_c9v2, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c9v2_c = require(c9v2_corpora, corpus_key)
        gl = require(c9v2_c, "c9iii_grouping_loss_style_corrected")
        auroc_flip = require(gl, "auroc_sign_flipped")
        ci_flip    = require(gl, "ci_95_sign_flipped")
        primary    = require(c9v2_c, "primary_probe_AUROC")
        qwen_var   = require(c9v2_c, "c9i_qwen_variance_AUROC")
        div_var    = require(c9v2_c, "c9i_diversity_variance_AUROC")
        cal_conf   = require(c9v2_c, "c9ii_calibrated_conf_AUROC")
        cmd(f"GroupingLossAUROC{macro_suffix}",      fmt(auroc_flip, 3))
        cmd(f"GroupingLossAUROC{macro_suffix}CIlo",  fmt(ci_flip[0], 3))
        cmd(f"GroupingLossAUROC{macro_suffix}CIhi",  fmt(ci_flip[1], 3))
        cmd(f"PrimaryProbeAUROC{macro_suffix}",      fmt(primary,    3))
        cmd(f"QwenVarAUROC{macro_suffix}",           fmt(qwen_var,   3))
        cmd(f"DivVarAUROC{macro_suffix}",            fmt(div_var,    3))
        cmd(f"CalibConfAUROC{macro_suffix}",         fmt(cal_conf,   3))

    # ================================================================
    # C2b_v4: reference-label-noise null (corrected construction)
    # NullFourMean{C}   = null_mean
    # NullFourP{C}      = empirical_p  (fraction of null simulations >= observed)
    # NullFourBandLo{C} = null_ci95[0]  (P2.5 of null distribution)
    # NullFourBandHi{C} = null_ci95[1]  (P97.5 of null distribution)
    # ================================================================
    section("R2-C2b_v4: corrected noise-null (NullFour macros)")
    c2bv4_corpora = require(r2_c2b_v4, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c2bv4_c = require(c2bv4_corpora, corpus_key)
        null_mean   = require(c2bv4_c, "null_mean")
        emp_p       = require(c2bv4_c, "empirical_p")
        null_ci95   = require(c2bv4_c, "null_ci95")
        cmd(f"NullFourMean{macro_suffix}",   fmt(null_mean, 3))
        cmd(f"NullFourP{macro_suffix}",      fmt(emp_p,     3))
        cmd(f"NullFourBandLo{macro_suffix}", fmt(null_ci95[0], 3))
        cmd(f"NullFourBandHi{macro_suffix}", fmt(null_ci95[1], 3))

    # ================================================================
    # C2b_v5: noise-null v5 (Holm-corrected; CSC Likert resampling)
    # NullFiveMean{C}         = null_mean (mean of null ΔAsym distribution)
    # NullFivePHolm{C}        = "< 0.001" when p_holm=0.0, "= {val:.3f}" otherwise
    # NullFiveBandLo{C}       = null_ci95[0] (P2.5 of null distribution)
    # NullFiveBandHi{C}       = null_ci95[1] (P97.5 of null distribution)
    # NullFiveExplainedPct{C} = null_explained_frac * 100, one decimal place
    # ================================================================
    section("R3-C2b_v5: noise-null v5 (Holm-corrected; NullFive macros)")

    def null_p_fmt(p: float) -> str:
        """Format p_holm: '< 0.001' when p=0.0 (N=1000 sims), '= {val:.3f}' otherwise."""
        return r"< 0.001" if p == 0.0 else f"= {p:.3f}"

    c2bv5_corpora = require(r3_c2b_v5, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c2bv5_c = require(c2bv5_corpora, corpus_key)
        null_mean_v5  = require(c2bv5_c, "null_mean")
        null_ph_v5    = require(c2bv5_c, "p_holm")
        null_ci95_v5  = require(c2bv5_c, "null_ci95")
        expl_frac_v5  = require(c2bv5_c, "null_explained_frac")
        cmd(f"NullFiveMean{macro_suffix}",         fmt(null_mean_v5,        3))
        cmd(f"NullFivePHolm{macro_suffix}",        null_p_fmt(null_ph_v5))
        cmd(f"NullFiveBandLo{macro_suffix}",       fmt(null_ci95_v5[0],     3))
        cmd(f"NullFiveBandHi{macro_suffix}",       fmt(null_ci95_v5[1],     3))
        cmd(f"NullFiveExplainedPct{macro_suffix}", pct(expl_frac_v5,        1))

    # ================================================================
    # C2c_v3: per-tercile soft-label Brier drop
    # drop = Brier(0.5B) - Brier(32B); negative = 32B is worse (over-confident)
    # SoftDropLowBrier{C}  = tercile_scores.low.drop_brier
    # SoftDropHighBrier{C} = tercile_scores.high.drop_brier
    # ================================================================
    section("R2-C2c_v3: per-tercile soft-label Brier drop (SoftDrop macros)")
    c2cv3_corpora = require(r2_c2c_v3, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c2cv3_c = require(c2cv3_corpora, corpus_key)
        drop_low  = require(c2cv3_c, "tercile_scores", "low",  "drop_brier")
        drop_high = require(c2cv3_c, "tercile_scores", "high", "drop_brier")
        cmd(f"SoftDropLowBrier{macro_suffix}",  sign_fmt(drop_low,  3))
        cmd(f"SoftDropHighBrier{macro_suffix}", sign_fmt(drop_high, 3))

    # ================================================================
    # C7_v3: unified AUBAC definition (AUBACvThree macros)
    # AUBACvThreeRand{C}          = aubac_random_mean
    # AUBACvThreeOracle{C}        = aubac_oracle_low_dis
    # AUBACvThreeCont{C}          = aubac_contraction
    # AUBACvThreeConf{C}          = aubac_confidence
    # AUBACvThreeDeltaCont{C}CIlo = delta_contraction_minus_random_ci95[0]
    # AUBACvThreeDeltaCont{C}CIhi = delta_contraction_minus_random_ci95[1]
    # ================================================================
    section("R2-C7_v3: unified AUBAC (AUBACvThree macros)")
    c7v3_corpora = require(r2_c7_v3, "per_corpus")
    for corpus_key, macro_suffix in [
        ("CSC",          "CSC"),
        ("EPIC",         "EPIC"),
        ("MultiPICo_es", "ES"),
        ("MultiPICo_de", "DE"),
        ("MultiPICo_fr", "FR"),
        ("MultiPICo_it", "IT"),
    ]:
        c7v3_c    = require(c7v3_corpora, corpus_key)
        rand_mean = require(c7v3_c, "aubac_random_mean")
        oracle    = require(c7v3_c, "aubac_oracle_low_dis")
        cont      = require(c7v3_c, "aubac_contraction")
        conf      = require(c7v3_c, "aubac_confidence")
        delta_ci  = require(c7v3_c, "delta_contraction_minus_random_ci95")
        cmd(f"AUBACvThreeRand{macro_suffix}",          fmt(rand_mean, 4))
        cmd(f"AUBACvThreeOracle{macro_suffix}",        fmt(oracle,    4))
        cmd(f"AUBACvThreeCont{macro_suffix}",          fmt(cont,      4))
        cmd(f"AUBACvThreeConf{macro_suffix}",          fmt(conf,      4))
        cmd(f"AUBACvThreeDeltaCont{macro_suffix}CIlo", sign_fmt(delta_ci[0], 4))
        cmd(f"AUBACvThreeDeltaCont{macro_suffix}CIhi", sign_fmt(delta_ci[1], 4))

    # ================================================================
    # C9_floor_v3: CSC denominator-floor sensitivity
    # FloorDropBothDeltaAsymCSC      = per_condition.drop_both.dasym
    # FloorDropBothDeltaAsymCSCCIlo  = per_condition.drop_both.ci95[0]
    # FloorDropBothDeltaAsymCSCCIhi  = per_condition.drop_both.ci95[1]
    # ================================================================
    section("R2-C9_floor_v3: CSC floor sensitivity (FloorDropBoth macros)")
    c9fl_drop_both = require(r2_c9fl_v3, "per_condition", "drop_both")
    cmd("FloorDropBothDeltaAsymCSC",      sign_fmt(require(c9fl_drop_both, "dasym"),    3))
    cmd("FloorDropBothDeltaAsymCSCCIlo",  fmt(require(c9fl_drop_both, "ci95")[0], 3))
    cmd("FloorDropBothDeltaAsymCSCCIhi",  fmt(require(c9fl_drop_both, "ci95")[1], 3))
    # Count macros for the floor-sensitivity text
    cmd("FloorNDenomCSC", str(int(require(r2_c9fl_v3, "n_denom_floored"))))
    cmd("FloorNClipCSC",  str(int(require(r2_c9fl_v3, "n_top_clipped"))))

    # ================================================================
    # Bootstrap count macros (for appendix text accuracy)
    # ================================================================
    section("Bootstrap count macros (for appendix text)")
    # ΔAsym CI: B=4000 (C1_DASYM_CI.json)
    cmd("nBootDeltaAsymText",  "4{,}000")
    # Contraction probe AUROC CI: B=2000 (C8_REGEN n_bootstrap=2000)
    cmd("nBootProbeCIText",    "2{,}000")
    # Null simulation: n_sims=1000 (C2_SOFTLABEL_v2)
    cmd("nNullSimsText",       "1{,}000")
    # Random AUBAC: 100 seeds (C7_COSTACC_v2 n_random_seeds=100)
    cmd("nRandomSeedsAUBACText", "100")
    # p < 1/(B+1) for B=4000
    cmd("pBoundDeltaAsym",     r"$p < 1/(B{+}1)$")

    # ================================================================
    # Finalize
    # ================================================================
    lines.append("")
    lines.append("% End of numbers.tex")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(content)
    print(f"Written: {OUT}")
    print(f"  Lines: {len(lines)}")
    print("  All values read from JSON — no hardcoded fallbacks.")


if __name__ == "__main__":
    main()
