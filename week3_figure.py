"""Week 3 results figure for the quad chart."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddddd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "figure.facecolor": "white",
    "axes.facecolor": "white",
})

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.0))


def clean(a, ylab=None):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(length=0)
    a.grid(axis="y", color=GRID, lw=0.7, alpha=0.8)
    a.set_axisbelow(True)
    if ylab:
        a.set_ylabel(ylab, fontsize=8.5)


# --- A: evaluation count is set by the constraint formulation ---------
a = ax[0, 0]
labels = ["102 constraints\nftol 1e-6", "2 constraints\nftol 1e-6",
          "2 constraints\nftol 1e-3"]
vals = [32, 12, 6]
cols = [BLUE, BLUE, ORANGE]
bars = a.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    a.text(b.get_x() + b.get_width()/2, v + 0.9, str(v),
           ha="center", fontsize=11, fontweight="bold", color=INK)
a.axhline(6, color=ORANGE, lw=1.4, ls="--", zorder=1)
a.text(-0.42, 7.0, "Kavin reported 6", fontsize=8.5, color=ORANGE,
       ha="left", style="italic")
a.set_ylim(0, 40)
clean(a, "SLSQP function evaluations")
a.set_title("A.  Evaluation count is set by the constraint\n"
            "formulation, not by anything fundamental",
            fontsize=10, fontweight="bold", loc="left")

# --- B: burn attribution ---------------------------------------------
a = ax[0, 1]
names = ["Trajectory\neffect", "Propellant\nmass effect", "Net change\n(both)"]
d = [0.00104, -0.00116, -0.00012]
cols = [ORANGE if x > 0 else BLUE for x in d]
bars = a.bar(names, d, color=cols, width=0.55)
for b, v in zip(bars, d):
    if abs(v) < 0.0005:                    # tiny bar: label above the axis
        ytxt, va = 0.00012, "bottom"
    elif v > 0:
        ytxt, va = v + 0.00006, "bottom"
    else:
        ytxt, va = v - 0.00006, "top"
    a.text(b.get_x() + b.get_width()/2, ytxt, f"{v:+.5f}",
           ha="center", va=va, fontsize=9.5, fontweight="bold", color=INK)
a.axhline(0, color=MUTED, lw=1.0)
a.set_ylim(-0.0016, 0.0016)
clean(a, "change in objective")
a.set_title("B.  The two effects of a shorter burn nearly\n"
            "cancel, which hides both if not separated",
            fontsize=10, fontweight="bold", loc="left")

# --- C: the flat direction -------------------------------------------
a = ax[1, 0]
names = ["Delivered\nimpulse", "Burn\ndelta-V", "Vehicle\nradius", "Objective"]
pct = [-86.6, -86.6, 0.39, 0.03]
cols = [ORANGE, ORANGE, BLUE, BLUE]
bars = a.barh(names[::-1], pct[::-1], color=cols[::-1], height=0.55)
for b, v in zip(bars, pct[::-1]):
    off = 3 if v > -10 else -3
    ha = "left" if v > -10 else "right"
    a.text(v + off, b.get_y() + b.get_height()/2, f"{v:+.2f}%",
           va="center", ha=ha, fontsize=9.5, fontweight="bold", color=INK)
a.axvline(0, color=MUTED, lw=1.0)
a.set_xlim(-110, 40)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
a.tick_params(length=0)
a.grid(axis="x", color=GRID, lw=0.7, alpha=0.8)
a.set_axisbelow(True)
a.set_xlabel("change from 5-step to 1-step burn", fontsize=8.5)
a.set_title("C.  Cutting the burn changes delivered impulse by 87%\n"
            "and the objective by 0.03%: a flat direction",
            fontsize=10, fontweight="bold", loc="left")

# --- D: compile vs solve ----------------------------------------------
a = ax[1, 1]
names = ["loop = False\n(RK4 unrolled)", "loop = True\n(RK4 looped)"]
comp = [11.99, 6.54]
solv = [0.49, 4.20]
x = np.arange(2)
b1 = a.bar(x, comp, width=0.5, color=BLUE, label="compile")
b2 = a.bar(x, solv, width=0.5, bottom=np.array(comp) + 0.12,
           color=AQUA, label="solve")
for i in range(2):
    a.text(i, comp[i]/2, f"{comp[i]:.1f}s", ha="center", va="center",
           fontsize=9.5, fontweight="bold", color="white")
    a.text(i, comp[i] + 0.12 + solv[i]/2 + 0.25, f"{solv[i]:.1f}s",
           ha="center", va="center", fontsize=9, fontweight="bold", color=INK)
    a.text(i, comp[i] + solv[i] + 1.1, f"total {comp[i]+solv[i]:.1f}s",
           ha="center", fontsize=8.5, color=MUTED)
a.set_xticks(x); a.set_xticklabels(names)
a.set_ylim(0, 17)
a.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
clean(a, "wall-clock time [s]")
a.set_title("D.  Both runs total ~12s, not the ~235s reported.\n"
            "Compilation, not optimization, dominates either way",
            fontsize=10, fontweight="bold", loc="left")

plt.tight_layout(pad=1.4, h_pad=2.6, w_pad=2.4)
plt.savefig("figures/week3_results.png", dpi=170,
            bbox_inches="tight", facecolor="white")
print("saved")
