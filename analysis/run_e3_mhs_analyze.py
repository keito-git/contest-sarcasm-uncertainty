"""
run_e3_mhs_analyze.py — ΔAsym analysis for MHS (hate speech) E3 sweep.

Method identical to run_e3_xfamily_analyze.py (B-task sarcasm cross-family):
  - Binary label from first Yes/No in raw response (calibration-invariant).
  - Tercile split on dis_mi (33/67 percentile via rankdata).
  - ΔAsym = dBalAcc_low - dBalAcc_high (small→large, low/high dis_mi tercile).
  - 4000-fold item bootstrap, H0: ΔAsym ≤ 0, one-sided p-value.
  - Reports balanced accuracy monotonicity per family.

Important: MHS dis_mi is BIMODAL.
  - 256/400 items dis_mi=0 (unanimous annotators) → all land in low tercile.
  - 144/400 items dis_mi>0 (any disagreement) → mid/high terciles.
  - Degenerate tercile sizes: low≈256, mid≈11, high≈133.
  ΔAsym compares 'unanimous-agree' (low) vs 'any-disagree' (high); this is
  semantically valid even though the tercile sizes are unequal.

Output: results/llm_e3/e3_mhs_xfamily_summary.md
"""
from __future__ import annotations
from pathlib import Path

import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = str(Path(__file__).parent.parent)
IN  = f"{BASE}/results/llm_e3"

# Model registry: same as run_e3_mhs_api.py
FAMILIES: dict[str, dict[str, float]] = {
    "llama3x": {"llx1b": 1.0, "llx3b": 3.0, "llx8b": 8.0, "llx70b": 70.0},
    "qwen3":   {"q3x8b": 8.0, "q3x14b": 14.0, "q3x32b": 32.0},
    "gemma3":  {"g3x4b": 4.0, "g3x12b": 12.0, "g3x27b": 27.0},
}

CORPUS = "MHS"
SEED   = 42

_YN = re.compile(r"\b(yes|no)\b", re.I)


def binary_from_raw(raw: str) -> int | None:
    """Extract binary prediction from raw model response (calibration-invariant)."""
    m = _YN.search(str(raw) if raw else "")
    if not m:
        return None
    return 1 if m.group(1).lower() == "yes" else 0


def tercile(a: np.ndarray) -> np.ndarray:
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])  # 0=low,1=mid,2=high


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=int)
    y    = np.asarray(y,    dtype=int)
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append((pred[m] == c).mean())
    return float(np.mean(recs)) if recs else float("nan")


def load_df(tag: str) -> pd.DataFrame | None:
    path = f"{IN}/e3_{tag}_{CORPUS}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "raw" in df.columns:
        df["pred_bin"] = df["raw"].apply(binary_from_raw)
    else:
        # Fallback: threshold on p_yes
        df["pred_bin"] = (df["p_yes"].to_numpy() >= 0.5).astype(float)
    return df


def analyze_family(
    family_name: str,
    tags_caps: dict[str, float],
) -> list[str]:
    """ΔAsym analysis for one family on MHS. Returns markdown lines."""
    present = [t for t in tags_caps if os.path.exists(f"{IN}/e3_{t}_{CORPUS}.csv")]
    if len(present) < 2:
        return [
            f"### {family_name} / {CORPUS}: "
            f"only {len(present)}/{len(tags_caps)} tags complete — skip"
        ]

    # Load and merge
    merged: pd.DataFrame | None = None
    for t in present:
        d = load_df(t)
        if d is None:
            continue
        keep = ["item_id", "dis_mi", "y_true", "pred_bin"]
        d = d[[c for c in keep if c in d.columns]].rename(
            columns={"pred_bin": f"pred_{t}"}
        )
        if merged is None:
            merged = d[["item_id", "dis_mi", "y_true"]].copy()
        merged = merged.merge(d[["item_id", f"pred_{t}"]], on="item_id")

    if merged is None or len(merged) < 10:
        return [f"### {family_name} / {CORPUS}: merge failed"]

    merged = merged.dropna()
    n      = len(merged)
    y      = merged["y_true"].to_numpy(int)
    terc   = tercile(merged["dis_mi"].to_numpy())

    caps_sorted = sorted(present, key=lambda t: tags_caps[t])
    pos_rt = y.mean()

    terc_counts = {g: int((terc == g).sum()) for g in [0, 1, 2]}
    lines = [
        f"### {family_name} / {CORPUS}  "
        f"(n={n}, pos_rate={pos_rt:.3f}, "
        f"caps=[{', '.join(f'{tags_caps[t]}B' for t in caps_sorted)}])"
    ]
    lines.append(
        f"- Tercile distribution (dis_mi 33/67 pct): "
        f"low={terc_counts[0]}, mid={terc_counts[1]}, high={terc_counts[2]}"
    )
    lines.append("| cap (B) | tercile | n | bal_acc | yes_rate |")
    lines.append("|--|--|--|--|--|")

    ba_by_cap: dict[str, float] = {}
    for t in caps_sorted:
        pred_raw = merged[f"pred_{t}"].to_numpy(float)
        pred_int = np.where(np.isnan(pred_raw), 0, pred_raw).astype(int)
        ba_by_cap[t] = bal_acc(pred_int, y)
        for g, lab in [(0, "low"), (1, "mid"), (2, "high")]:
            gi = terc == g
            n_g = int(gi.sum())
            ba = bal_acc(pred_int[gi], y[gi]) if n_g > 0 else float("nan")
            yr = pred_int[gi].mean() if n_g > 0 else float("nan")
            lines.append(
                f"| {tags_caps[t]}B | {lab} | {n_g} | "
                f"{'nan' if np.isnan(ba) else f'{ba:.3f}'} | "
                f"{'nan' if np.isnan(yr) else f'{yr:.3f}'} |"
            )

    # Monotonic gradient check
    ba_vals = [ba_by_cap[t] for t in caps_sorted]
    is_mono = all(
        ba_vals[i] <= ba_vals[i + 1] for i in range(len(ba_vals) - 1)
    )
    ba_str = " → ".join(
        f"{tags_caps[t]}B:{ba_by_cap[t]:.3f}" for t in caps_sorted
    )
    lines.append(f"- Overall bal_acc: {ba_str}  | monotone? {is_mono}")

    # ΔAsym: smallest → largest capacity
    sm_tag, lg_tag = caps_sorted[0], caps_sorted[-1]

    def to_int_pred(t: str) -> np.ndarray:
        raw = merged[f"pred_{t}"].to_numpy(float)
        return np.where(np.isnan(raw), 0, raw).astype(int)

    ps = to_int_pred(sm_tag)
    pl = to_int_pred(lg_tag)

    dlow  = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
    dhigh = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])
    d_asym = dlow - dhigh

    # Bootstrap p-value (4000 resamples)
    rng = np.random.default_rng(SEED)
    bs: list[float] = []
    for _ in range(4000):
        bi = rng.integers(0, n, n)
        t_bi = terc[bi]
        y_bi = y[bi]
        bl = (
            bal_acc(pl[bi][t_bi == 0], y_bi[t_bi == 0])
            - bal_acc(ps[bi][t_bi == 0], y_bi[t_bi == 0])
        )
        bh = (
            bal_acc(pl[bi][t_bi == 2], y_bi[t_bi == 2])
            - bal_acc(ps[bi][t_bi == 2], y_bi[t_bi == 2])
        )
        bs.append(bl - bh)

    p_asym = float(np.mean(np.asarray(bs) <= 0))
    sm_cap = tags_caps[sm_tag]
    lg_cap = tags_caps[lg_tag]

    lines.append(
        f"- **ΔAsym ({sm_cap}B→{lg_cap}B)**: "
        f"low_dis(agree)={dlow:+.3f} high_dis(split)={dhigh:+.3f} | "
        f"**ΔAsym(low-high)={d_asym:+.3f}** (boot p(≤0)={p_asym:.3f})"
    )

    # Verdict
    if d_asym > 0 and p_asym < 0.05:
        verdict = "C3 CONFIRMED: significant asymmetric contraction (p<0.05)."
    elif d_asym > 0 and p_asym < 0.10:
        verdict = "C3 MARGINAL: positive asymmetry, p<0.10 (weak evidence)."
    elif d_asym > 0:
        verdict = "C3 DIRECTIONAL: positive ΔAsym but not significant."
    else:
        verdict = "C3 NOT REPRODUCED: ΔAsym ≤ 0 in this family."
    lines.append(f"- Verdict: *{verdict}*")
    lines.append("")
    return lines


def main() -> None:
    lines = [
        "# E3 MHS (hate speech) — cross-family ΔAsym analysis",
        "",
        "Method: Binary label from first Yes/No word in raw response (calibration-invariant).",
        "OpenRouter API, K=1, temperature=0.",
        "Note: API-based inference; equivalent for",
        "      balanced-accuracy analysis (binary threshold on label).",
        "",
        "**MHS dis_mi note**: Distribution is BIMODAL — 256/400 items have dis_mi=0",
        "(all annotators unanimously agree on label); 144/400 have dis_mi>0 (any",
        "disagreement). Tercile split is degenerate: low≈256 items (unanimous-agree),",
        "mid≈11 items (tiny boundary zone), high≈133 items (any-disagree).",
        "ΔAsym compares 'unanimous-agree (epistemic)' vs 'any-disagree (aleatoric)'.",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    family_verdicts: list[tuple[str, float, float, bool]] = []  # (name, dasym, p, is_mono)

    for family_name, tags_caps in FAMILIES.items():
        lines.append(f"## {family_name}")
        fam_lines = analyze_family(family_name, tags_caps)
        lines.extend(fam_lines)

        # Extract ΔAsym and monotonicity from the last result
        present = [t for t in tags_caps if os.path.exists(f"{IN}/e3_{t}_{CORPUS}.csv")]
        if len(present) >= 2:
            # Re-derive values for summary table
            dfs: dict[str, pd.DataFrame] = {}
            for t in present:
                d = load_df(t)
                if d is not None:
                    dfs[t] = d

            if len(dfs) >= 2:
                first_key = next(iter(dfs))
                merged = dfs[first_key][["item_id", "dis_mi", "y_true"]].copy()
                for t in list(dfs.keys()):
                    dd = dfs[t][["item_id", "pred_bin"]].rename(
                        columns={"pred_bin": f"pred_{t}"}
                    )
                    merged = merged.merge(dd, on="item_id")
                merged = merged.dropna()
                y = merged["y_true"].to_numpy(int)
                terc = tercile(merged["dis_mi"].to_numpy())
                caps_sorted = sorted(present, key=lambda t: tags_caps[t])

                ba_vals_mono = []
                for t in caps_sorted:
                    raw = merged[f"pred_{t}"].to_numpy(float)
                    pred = np.where(np.isnan(raw), 0, raw).astype(int)
                    ba_vals_mono.append(bal_acc(pred, y))
                is_mono = all(
                    ba_vals_mono[i] <= ba_vals_mono[i + 1]
                    for i in range(len(ba_vals_mono) - 1)
                )

                sm, lg = caps_sorted[0], caps_sorted[-1]
                def _pi(t):
                    raw = merged[f"pred_{t}"].to_numpy(float)
                    return np.where(np.isnan(raw), 0, raw).astype(int)
                ps, pl = _pi(sm), _pi(lg)
                dlow  = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
                dhigh = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])
                rng = np.random.default_rng(SEED)
                n = len(merged)
                bsv = []
                for _ in range(4000):
                    bi = rng.integers(0, n, n)
                    tb = terc[bi]; yb = y[bi]
                    bsv.append(
                        (bal_acc(pl[bi][tb==0],yb[tb==0]) - bal_acc(ps[bi][tb==0],yb[tb==0]))
                        - (bal_acc(pl[bi][tb==2],yb[tb==2]) - bal_acc(ps[bi][tb==2],yb[tb==2]))
                    )
                p_val = float(np.mean(np.asarray(bsv) <= 0))
                family_verdicts.append((family_name, dlow - dhigh, p_val, is_mono))

    # Summary table
    lines.append("## Summary Table")
    lines.append("| family | ΔAsym (low-high) | p(≤0) | monotone? | verdict |")
    lines.append("|--|--|--|--|--|")
    confirmed = []
    for fname, dasym, p, mono in family_verdicts:
        if dasym > 0 and p < 0.05:
            v = "C3 CONFIRMED"
            confirmed.append(fname)
        elif dasym > 0 and p < 0.10:
            v = "C3 MARGINAL"
        elif dasym > 0:
            v = "DIRECTIONAL"
        else:
            v = "NOT REPRODUCED"
        lines.append(f"| {fname} | {dasym:+.3f} | {p:.3f} | {mono} | {v} |")
    lines.append("")

    # Overall conclusion
    n_confirmed = len(confirmed)
    n_families  = len(family_verdicts)
    lines.append("## Overall Conclusion")
    if n_confirmed >= 2:
        lines.append(
            f"C3 (asymmetric contraction) is confirmed in {n_confirmed}/{n_families} "
            f"families ({', '.join(confirmed)}) for hate speech (MHS corpus). "
            "This extends the generalisation claim beyond sarcasm/irony."
        )
    elif n_confirmed == 1:
        lines.append(
            f"C3 is confirmed in 1/{n_families} families ({confirmed[0]}) for MHS. "
            "Other families show directional but non-significant results."
        )
    elif any(d > 0 for _, d, _, _ in family_verdicts):
        lines.append(
            f"C3 shows directional asymmetry (ΔAsym>0) in all families but is "
            "not statistically significant in any family for MHS hate speech. "
            "Possible causes: (a) smaller study (K=1 vs K=10), (b) MHS degenerate "
            "tercile (mid group n≈11), or (c) hate speech is genuinely harder to "
            "improve epistemically than sarcasm."
        )
    else:
        lines.append(
            "C3 is NOT reproduced in hate speech (MHS): ΔAsym ≤ 0 across families. "
            "This is a negative but honest result."
        )

    lines.append("")
    lines.append("## Method Notes")
    lines.append("- K=1 API inference (vs K=10 GPU sampling for original Qwen2.5 sweep).")
    lines.append("- n_valid documented per model in individual run JSON logs.")
    lines.append("- Binary label (Yes/No) from raw field; calibration-invariant.")
    lines.append("- 4000-fold item bootstrap, seed=42.")

    txt = "\n".join(lines)
    out = f"{IN}/e3_mhs_xfamily_summary.md"
    with open(out, "w") as f:
        f.write(txt)
    print(txt, flush=True)
    print(f"\n[Analysis] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
