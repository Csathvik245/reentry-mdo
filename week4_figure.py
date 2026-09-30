"""Week 4 results figure."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from newtonian_ref import (coefficients, closed_form_CA, cp_max,
                           theta_c_for_spherical_segment, meridian)
import reference_model as rm

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
CPMAX = cp_max()
TH = theta_c_for_spherical_segment(2.4, 1.0)

fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))


def clean(a, ylab=None, xlab=None):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(length=0)
    a.grid(axis="y", color=GRID, lw=0.7, alpha=0.8)
    a.set_axisbelow(True)
    if ylab:
        a.set_ylabel(ylab, fontsize=8.5)
    if xlab:
        a.set_xlabel(xlab, fontsize=8.5)


# --- A: panel model converges to the closed form ----------------------
a = ax[0, 0]
ns = [10, 20, 40, 80, 160, 320]
exact = closed_form_CA(2.4, TH, 1.0)
errs = []
for n in ns:
    c = coefficients(2.4, TH, 1.0, n_cap=n, n_cone=0, n_theta=4 * n)
    errs.append(abs(c['C_D'] - exact) / exact)
npan = [n * 4 * n for n in ns]
a.loglog(npan, errs, "o-", color=BLUE, lw=2, ms=6, label="panel model")
ref = np.array(errs[0]) * (np.array(npan) / npan[0]) ** -1.0
a.loglog(npan, ref, "--", color=MUTED, lw=1.2, label="2nd order (slope $-1$)")
a.set_xlabel("number of panels", fontsize=8.5)
a.set_ylabel("relative error in $C_D$", fontsize=8.5)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
a.tick_params(length=0)
a.grid(color=GRID, lw=0.7, alpha=0.8, which="both")
a.set_axisbelow(True)
a.legend(frameon=False, fontsize=8)
a.set_title("A.  Panel model converges to the closed form\n"
            "at second order, to 4 parts in $10^6$",
            fontsize=10, fontweight="bold", loc="left")

# --- B: C_D against the Newtonian ceiling -----------------------------
a = ax[0, 1]
labels = ["45° sphere\ncone", "hemisphere", "70° sphere\ncone",
          "baseline\nshape", "flat plate\n(ceiling)", "value in\nuse"]
vals = [closed_form_CA(0.1, 45.0, 1.0), CPMAX / 2,
        closed_form_CA(0.5, 70.0, 1.0), exact, CPMAX, 6.14]
cols = [BLUE] * 4 + [MUTED, ORANGE]
bars = a.bar(labels, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    a.text(b.get_x() + b.get_width() / 2, v + 0.13, f"{v:.2f}",
           ha="center", fontsize=9.5, fontweight="bold", color=INK)
a.axhline(CPMAX, color=MUTED, lw=1.3, ls="--", zorder=1)
a.text(-0.45, CPMAX + 0.16, "Newtonian ceiling $C_{p,max}=1.84$",
       fontsize=8, color=MUTED, style="italic", ha="left")
a.set_ylim(0, 7.4)
clean(a, "$C_D$ referenced to frontal area")
a.set_title("B.  No convex body can exceed $C_{p,max}$.\n"
            "The value in use sits 3.3× above it",
            fontsize=10, fontweight="bold", loc="left")

# --- C: the corner in the parameterization ----------------------------
a = ax[1, 0]
rn = np.linspace(2.28, 2.52, 121)
cd = [coefficients(r, TH, 1.0, alpha_deg=10.0,
                   n_cap=30, n_cone=15, n_theta=60)['C_D'] for r in rn]
a.plot(rn, cd, "-", color=BLUE, lw=2)
a.axvline(2.4, color=ORANGE, lw=1.3, ls="--")
i = np.searchsorted(rn, 2.4)
a.plot([2.4], [cd[i]], "o", color=ORANGE, ms=7, zorder=5)
a.annotate("frustum length\nreaches zero here",
           xy=(2.4, cd[i]), xytext=(2.428, cd[i] + 0.055),
           fontsize=8, color=ORANGE, ha="left",
           arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1))
a.text(2.285, cd[i] + 0.10,
       "one-sided slopes\n0.12 vs 1.34", fontsize=8, color=INK)
clean(a, "$C_D$", "nose radius $R_n$  (with $R_b=1$, $\\theta_c$ fixed)")
a.set_title("C.  The shape family has a corner, not the AD.\n"
            "AD returns the right one-sided derivative",
            fontsize=10, fontweight="bold", loc="left")

# --- D: trajectory effect of the corrected C_D ------------------------
a = ax[1, 1]
for cd_val, col, lab in [(6.14, ORANGE, "$C_D=6.14$ (in use)"),
                         (exact, BLUE, f"$C_D={exact:.2f}$ (Newtonian)")]:
    rm.C_D = cd_val
    s = rm.simulate(gamma0=-0.1187, Fn=521.0, Fv=31050.8, R=5.9433)
    a.plot(s["t"], s["q_coded"] / 1e-5, lw=2, color=col, label=lab)
rm.C_D = 6.14
a.axhline(8.0, color=MUTED, lw=1.3, ls="--")
a.text(99, 8.35, "constraint  $8\\times10^{-5}$", fontsize=8, color=MUTED,
       ha="right", style="italic")
a.set_ylim(0, 10.2)
a.legend(frameon=False, fontsize=8.5, loc="upper left")
clean(a, "heat flux  [$10^{-5}$, model units]", "time [s]")
a.set_title("D.  Correcting $C_D$ pushes peak heating 2× higher,\n"
            "past the constraint: the current optimum goes infeasible",
            fontsize=10, fontweight="bold", loc="left")

plt.tight_layout(pad=1.4, h_pad=2.8, w_pad=2.4)
plt.savefig("figures/week4_results.png", dpi=170, bbox_inches="tight",
            facecolor="white")
print("saved figures/week4_results.png")
