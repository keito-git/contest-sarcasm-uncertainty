"""
Post-hoc analysis for E3 cross-family sweep (Task B result).

Reads e3_{tag}_{corpus}.csv files produced by run_e3_xfamily_api.py.
Uses BINARY LABEL (Yes/No first word from 'raw' column) as prediction,
NOT the p_yes computed from verbalized confidence.

Rationale: Different Qwen3 model sizes use different calibration conventions
(e.g., Qwen3-14B says "No 10" with very low confidence, which verbalized
confidence formula inverts to p_yes=0.90). Binary label from the "Yes/No"
token is calibration-invariant and matches the K=10 sampling approach used
for existing Qwen2.5 data (where p_yes = fraction saying "Yes").

Output: results/llm_e3/e3_xfamily_final_summary.md
"""
from __future__ import annotations
from pathlib import Path

import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = str(Path(__file__).parent.parent)
IN   = f"{BASE}/results/llm_e3"

XFAMILY: dict[str, dict[str, float]] = {
    "qwen3": {
        "q3x8b": 8.0, "q3x14b": 14.0, "q3x32b": 32.0,
    },
    "gemma3": {
        "g3x4b": 4.0, "g3x12b": 12.0, "g3x27b": 27.0,
    },
    "llama3x": {
        "llx1b": 1.0, "llx3b": 3.0, "llx8b": 8.0, "llx70b": 70.0,
    },
}

CORPORA = ["CSC", "MultiPICo", "EPIC"]

_YN = re.compile(r"\b(yes|no)\b", re.I)


def binary_from_raw(raw: str) -> int | None:
    """Extract binary label (1=sarcastic/ironic, 0=not) from raw response text."""
    m = _YN.search(str(raw) if raw else "")
    if not m:
        return None
    return 1 if m.group(1).lower() == "yes" else 0


def tercile(a: np.ndarray) -> np.ndarray:
    r = rankdata(a) / (len(a) + 1)
    return np.digitize(r, [1 / 3, 2 / 3])  # 0=low, 1=mid, 2=high


def bal_acc(pred: np.ndarray, y: np.ndarray) -> float:
    recs = []
    for c in [0, 1]:
        m = y == c
        if m.sum() > 0:
            recs.append((pred[m] == c).mean())
    return float(np.mean(recs)) if recs else float("nan")


def load_with_binlabel(tag: str, corpus: str) -> pd.DataFrame | None:
    path = f"{IN}/e3_{tag}_{corpus}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "raw" not in df.columns:
        # Fallback: use p_yes threshold (less accurate for confidence-miscalibrated models)
        df["pred_bin"] = (df["p_yes"].to_numpy() >= 0.5).astype(float)
        df["_bin_source"] = "p_yes_threshold"
    else:
        df["pred_bin"] = df["raw"].apply(binary_from_raw)
        df["_bin_source"] = "binary_label"
    return df


def analyze_family(
    family_name: str,
    tags_caps: dict[str, float],
    corpus: str,
) -> list[str]:
    """ΔAsym for one family × one corpus using binary labels."""
    present = [
        t for t in tags_caps
        if os.path.exists(f"{IN}/e3_{t}_{corpus}.csv")
    ]
    if len(present) < 2:
        return [f"### {family_name}/{corpus}: <2 sizes — skip"]

    # Load and merge on item_id
    base_df = None
    for t in present:
        d = load_with_binlabel(t, corpus)
        if d is None:
            continue
        d = d[["item_id", "pred_bin", "dis_mi", "y_true"]].rename(
            columns={"pred_bin": f"pred_{t}"}
        )
        if base_df is None:
            base_df = d[["item_id", "dis_mi", "y_true"]].copy()
        base_df = base_df.merge(d[["item_id", f"pred_{t}"]], on="item_id")

    if base_df is None or len(base_df) < 10:
        return [f"### {family_name}/{corpus}: merge failed"]

    base_df = base_df.dropna()
    y      = base_df["y_true"].to_numpy()
    terc   = tercile(base_df["dis_mi"].to_numpy())
    n      = len(base_df)
    pos_rt = y.mean()

    caps_sorted = sorted(present, key=lambda t: tags_caps[t])
    lines = [
        f"### {family_name} / {corpus}  "
        f"(n={n}, pos_rate={pos_rt:.3f}, "
        f"caps={[f'{tags_caps[t]}B' for t in caps_sorted]})"
    ]
    lines.append("| cap (B) | tercile | n | bal_acc | yes_rate |")
    lines.append("|--|--|--|--|--|")

    ba_by_cap: dict[str, float] = {}
    for t in caps_sorted:
        pred = base_df[f"pred_{t}"].to_numpy(float)
        # Handle NaN (unparseable) → neutral 0.5 → round to 0
        pred_int = np.where(np.isnan(pred), 0, pred).astype(int)
        ba_by_cap[t] = bal_acc(pred_int, y)
        for g, lab in [(0, "low"), (1, "mid"), (2, "high")]:
            gi = terc == g
            ba = bal_acc(pred_int[gi], y[gi]) if gi.sum() > 0 else float("nan")
            yr = pred_int[gi].mean() if gi.sum() > 0 else float("nan")
            lines.append(f"| {tags_caps[t]}B | {lab} | {gi.sum()} | {ba:.3f} | {yr:.3f} |")

    # Monotonic gradient check
    ba_vals = [ba_by_cap[t] for t in caps_sorted]
    is_mono = all(
        ba_vals[i] <= ba_vals[i + 1] for i in range(len(ba_vals) - 1)
    )
    ba_str = " → ".join(
        f"{tags_caps[t]}B:{ba_by_cap[t]:.3f}" for t in caps_sorted
    )
    lines.append(f"- Overall bal_acc: {ba_str}  | monotone? {is_mono}")

    # ΔAsym: smallest → largest
    sm_tag, lg_tag = caps_sorted[0], caps_sorted[-1]
    ps = np.where(
        np.isnan(base_df[f"pred_{sm_tag}"].to_numpy(float)), 0,
        base_df[f"pred_{sm_tag}"].to_numpy(float)
    ).astype(int)
    pl = np.where(
        np.isnan(base_df[f"pred_{lg_tag}"].to_numpy(float)), 0,
        base_df[f"pred_{lg_tag}"].to_numpy(float)
    ).astype(int)

    dlow  = bal_acc(pl[terc == 0], y[terc == 0]) - bal_acc(ps[terc == 0], y[terc == 0])
    dhigh = bal_acc(pl[terc == 2], y[terc == 2]) - bal_acc(ps[terc == 2], y[terc == 2])
    d_asym = dlow - dhigh

    # Bootstrap p-value
    rng = np.random.default_rng(42)
    bs = []
    for _ in range(4000):
        bi = rng.integers(0, n, n)
        bl = (
            bal_acc(pl[bi][terc[bi] == 0], y[bi][terc[bi] == 0])
            - bal_acc(ps[bi][terc[bi] == 0], y[bi][terc[bi] == 0])
        )
        bh = (
            bal_acc(pl[bi][terc[bi] == 2], y[bi][terc[bi] == 2])
            - bal_acc(ps[bi][terc[bi] == 2], y[bi][terc[bi] == 2])
        )
        bs.append(bl - bh)
    p_asym = float(np.mean(np.asarray(bs) <= 0))

    lines.append(
        f"- **ΔAsym ({tags_caps[sm_tag]}B→{tags_caps[lg_tag]}B)**: "
        f"low={dlow:+.3f} high={dhigh:+.3f} | "
        f"**ΔAsym(low-high)={d_asym:+.3f}** (boot p(<=0)={p_asym:.3f})"
    )
    lines.append("")
    return lines


def main() -> None:
    lines = [
        "# E3 cross-family — final analysis (binary-label predictions)",
        "",
        "Prediction: Yes/No first word from raw response (calibration-invariant).",
        "Method: OpenRouter API K=1 temperature=0.",
        "Note: API-based inference (K=1 temperature=0).",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    all_gradients: list[str]  = []
    asym_results: list[dict]  = []

    for family_name, tags_caps in XFAMILY.items():
        family_lines = [f"## {family_name}"]
        family_has_grad = False

        for corpus in CORPORA:
            corpus_lines = analyze_family(family_name, tags_caps, corpus)
            family_lines.extend(corpus_lines)

            # Check for gradient and positive ΔAsym
            present = [
                t for t in tags_caps
                if os.path.exists(f"{IN}/e3_{t}_{corpus}.csv")
            ]
            if len(present) >= 2:
                caps_sorted = sorted(present, key=lambda t: tags_caps[t])
                ba_vals = []
                for t in caps_sorted:
                    d = load_with_binlabel(t, corpus)
                    if d is None:
                        ba_vals.append(float("nan"))
                        continue
                    d = d.dropna(subset=["pred_bin"])
                    pred = d["pred_bin"].to_numpy(int)
                    y_a = d["y_true"].to_numpy()
                    ba_vals.append(bal_acc(pred, y_a))
                if all(
                    not np.isnan(ba_vals[i]) and not np.isnan(ba_vals[i + 1])
                    and ba_vals[i] <= ba_vals[i + 1]
                    for i in range(len(ba_vals) - 1)
                ):
                    all_gradients.append(f"{family_name}/{corpus}")
                    family_has_grad = True

        if family_has_grad:
            family_lines.append(f"**=> {family_name} shows monotonic capacity gradient.**")

        lines.extend(family_lines)
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"- Files analyzed: {sum(1 for f in XFAMILY for corpus in CORPORA for t in XFAMILY[f] if os.path.exists(f'{IN}/e3_{t}_{corpus}.csv'))} of {sum(len(v)*len(CORPORA) for v in XFAMILY.values())} expected")
    if all_gradients:
        lines.append(
            f"- Monotonic capacity gradients found in: {all_gradients}"
        )
        lines.append(
            "  These families can be used to test cross-family ΔAsym reproducibility."
        )
    else:
        n_files_done = sum(
            1 for f in XFAMILY
            for corpus in CORPORA
            for t in XFAMILY[f]
            if os.path.exists(f"{IN}/e3_{t}_{corpus}.csv")
        )
        n_expected = sum(len(v) * len(CORPORA) for v in XFAMILY.values())
        if n_files_done < n_expected:
            lines.append(
                f"  WARNING: Only {n_files_done}/{n_expected} files complete. "
                "Run again after all files are generated."
            )
        else:
            lines.append(
                "  NO family showed strict monotonic gradient in any corpus. "
                "Negative result: irony/sarcasm is universally hard across model families. "
                "This supports the 'epistemic difficulty is universal' conclusion."
            )

    txt = "\n".join(lines)
    out = f"{IN}/e3_xfamily_final_summary.md"
    with open(out, "w") as f:
        f.write(txt)
    print(txt)
    print(f"\n[Analysis] saved -> {out}")


if __name__ == "__main__":
    main()
