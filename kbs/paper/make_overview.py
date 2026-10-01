"""
Methodology overview figure (Fig. overview) for the KBS manuscript, drawn in code.
Content follows Sections 3 and 4 of the manuscript.
Layout (v3): the item x_i goes to human annotators and to the LLMs; the two sources are grouped in one
dashed frame ("item-level data") from which a single arrow fans out to the three stages. Each stage box
lists, as coloured tags, exactly which quantities it uses (blue = from human annotations,
green = from LLM outputs), so the arrows carry no hidden meaning.
  Stage 1: dis_i, U^s_i
  Stage 2: y_i, dis_i (terciles), Qwen2.5 0.5B / 32B majority-vote predictions y-hat (BalAcc)
  Stage 3: Qwen2.5 3B / 0.5B predictions (the probe); y_i only defines the evaluation target y_fix
Output: $CONTEST_PAPER_OUT/figures/fig_overview.pdf  (default: paper_out/figures/fig_overview.pdf)
Usage (from the repository root): python kbs/paper/make_overview.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "pdf.fonttype": 42,
})

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contest_paths import PAPER_OUT  # noqa: E402  (repo-root path config)

HERE = PAPER_OUT / "figures"
HERE.mkdir(parents=True, exist_ok=True)
W, H = 7.4, 3.6
C_ITEM, C_ANN, C_LLM, C_ST, C_OUT, C_EDGE = "#EEEEEE", "#D0E4F0", "#D6ECD2", "#FFFFFF", "#F8E0C8", "#333333"

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


PAD = 0.02                      # FancyBboxPatch pad: the drawn border lies PAD outside (x, y, w, h)
EDGE = PAD + 0.8 / 72 / 2      # plus half the border stroke (0.8 pt) in inches = data units


def L(x):
    """x-coordinate just outside the left border of a box whose nominal left edge is x."""
    return x - EDGE


def R(x):
    """x-coordinate just outside the right border of a box whose nominal right edge is x."""
    return x + EDGE


def rbox(x, y, w, h, fc, ls="-", lw=0.8, ec=C_EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, ls=ls))


LW = 0.9   # one line width for every connector


def connector(points, head=True):
    """Polyline through points drawn as one patch; arrow head at the last point if head."""
    path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>" if head else "-", mutation_scale=9, lw=LW,
                                 color=C_EDGE, shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))


def tags(xc, y, items, size=7.8, gap=0.06):
    """Row of small coloured tags centred at xc; items = [(text, colour), ...]."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts, widths = [], []
    for txt, col in items:
        t = ax.text(0, y, txt, fontsize=size, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none"))
        bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
        texts.append(t)
        widths.append(bb.width + 0.08)
    x = xc - (sum(widths) + gap * (len(items) - 1)) / 2
    for t, w in zip(texts, widths):
        t.set_x(x + 0.04)
        x += w + gap


# column headers
for xc, label in ((1.30, "Inputs"), (4.12, "Diagnostic stages"), (6.50, "Outputs")):
    ax.text(xc, H - 0.05, label, ha="center", va="top", fontsize=9.3, style="italic")

# ---- stage rows (needed first: the input group is centred on the Stage 2 row)
xs, ws, hs = 3.05, 2.15, 0.98
st_y = [2.42, 1.27, 0.12]
ymid = st_y[1] + hs / 2

# ---- inputs: item -> {annotations, LLM outputs}, grouped as item-level data
xi, wi, hi = 0.05, 0.62, 0.55
xsrc, wsrc, hsrc = 0.98, 1.50, 0.80
gap_src = 0.43                                  # vertical gap between the two source boxes
fw, fh = wsrc + 0.20, 2 * hsrc + gap_src + 0.12 + 0.30
fx, fy = xsrc - 0.10, ymid - fh / 2             # frame centre == Stage 2 centre
src_y = [fy + 0.12 + hsrc + gap_src, fy + 0.12]
rbox(fx, fy, fw, fh, "none", ls=(0, (3, 2)), lw=0.8, ec="#666666")
ax.text(fx + fw / 2, fy + fh - 0.07, "item-level data", ha="center", va="top", fontsize=8.2, style="italic",
        color="#444444")

item_y = (src_y[0] + src_y[1] + hsrc) / 2 - hi / 2
rbox(xi, item_y, wi, hi, C_ITEM)
ax.text(xi + wi / 2, item_y + hi / 2, "Item\n$x_i$", ha="center", va="center", fontsize=9.3, linespacing=1.2)
xm = xi + wi + 0.10
for y in src_y:
    connector([(R(xi + wi), item_y + hi / 2), (xm, item_y + hi / 2), (xm, y + hsrc / 2), (L(xsrc), y + hsrc / 2)])

rbox(xsrc, src_y[0], wsrc, hsrc, C_ANN)
ax.text(xsrc + wsrc / 2, src_y[0] + hsrc - 0.11, "Human annotations", ha="center", va="top", fontsize=9.1,
        fontweight="bold")
ax.text(xsrc + wsrc / 2, src_y[0] + 0.25, "$K_i$ labels $\\rightarrow$ $y_i$, $\\mathrm{dis}_i$",
        ha="center", va="center", fontsize=8.4)
rbox(xsrc, src_y[1], wsrc, hsrc, C_LLM)
ax.text(xsrc + wsrc / 2, src_y[1] + hsrc - 0.11, "LLM outputs", ha="center", va="top", fontsize=9.1,
        fontweight="bold")
ax.text(xsrc + wsrc / 2, src_y[1] + 0.25, "$U^{s}_i$; Qwen2.5 $\\hat{p}_i$, $\\hat{y}_i$ (0.5B$\\to$32B)",
        ha="center", va="center", fontsize=8.0)

# ---- one arrow from the grouped data (straight into Stage 2), branching to Stages 1 and 3
xb = fx + fw + 0.20
for y in st_y:
    connector([(R(fx + fw), ymid), (xb, ymid), (xb, y + hs / 2), (L(xs), y + hs / 2)])

STAGES = [
    ("Stage 1: single-signal audit", "Spearman $\\rho(U^{s}_i,\\ \\mathrm{dis}_i)$",
     [("$\\mathrm{dis}_i$", C_ANN), ("$U^{s}_i$", C_LLM)]),
    ("Stage 2: differential contraction",
     "$\\Delta\\mathrm{Asym}=\\Delta\\mathrm{BalAcc}_{\\mathrm{low}}-\\Delta\\mathrm{BalAcc}_{\\mathrm{high}}$",
     [("$y_i$", C_ANN), ("$\\mathrm{dis}_i$ terciles", C_ANN), ("$\\hat{y}^{0.5\\mathrm{B}}_i,\\hat{y}^{32\\mathrm{B}}_i$", C_LLM)]),
    ("Stage 3: contraction probe", "$\\mathrm{contr}_i=|\\hat{p}^{3\\mathrm{B}}_i-\\hat{p}^{0.5\\mathrm{B}}_i|$",
     [("$\\hat{p}^{3\\mathrm{B}}_i,\\hat{p}^{0.5\\mathrm{B}}_i$", C_LLM), ("$y_i$ (target $y_{\\mathrm{fix}}$ only)", C_ANN)]),
]
for (title, formula, used), y in zip(STAGES, st_y):
    rbox(xs, y, ws, hs, C_ST)
    ax.text(xs + ws / 2, y + hs - 0.10, title, ha="center", va="top", fontsize=9.1, fontweight="bold")
    ax.text(xs + ws / 2, y + hs / 2 - 0.02, formula, ha="center", va="center", fontsize=8.6)
    tags(xs + ws / 2, y + 0.17, used)

# ---- outputs
xo, wo, ho = 5.62, 1.72, 0.74
OUTS = ["Does a single signal\ntrack contestability?", "Do consensus items gain\nmore from scale?",
        "AUROC of a label-free\nescalation signal"]
for txt, y in zip(OUTS, st_y):
    oy = y + (hs - ho) / 2
    rbox(xo, oy, wo, ho, C_OUT)
    ax.text(xo + wo / 2, oy + ho / 2, txt, ha="center", va="center", fontsize=8.7, linespacing=1.35)
    connector([(R(xs + ws), y + hs / 2), (L(xo), oy + ho / 2)])

out = HERE / "fig_overview.pdf"
fig.savefig(out)
print("wrote", out)
