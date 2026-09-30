"""
Validation of the Newtonian panel model against the closed-form solution,
and resolution of the C_D = 6.14 reference-area question.
"""
import numpy as np
from newtonian_ref import (cp_max, coefficients, closed_form_CA,
                           theta_c_for_spherical_segment, panels)

CPMAX = cp_max()

print("=" * 78)
print("1. PANEL MODEL vs CLOSED FORM  (alpha = 0)")
print("=" * 78)
shapes = [
    ("hemisphere",                 1.0,  0.001, 1.0),
    ("70 deg sphere-cone, Rn/Rb=0.5", 0.5, 70.0, 1.0),
    ("60 deg sphere-cone, Rn/Rb=0.25", 0.25, 60.0, 1.0),
    ("45 deg sphere-cone, Rn/Rb=0.1", 0.1, 45.0, 1.0),
    ("Orion-like segment, Rn/Rb=2.4", 2.4, None, 1.0),
]
print(f"{'shape':<34}{'panel C_D':>12}{'closed form':>13}{'rel err':>11}")
for name, rn, th, rb in shapes:
    if th is None:
        th = theta_c_for_spherical_segment(rn, rb)
    c = coefficients(rn, th, rb, n_cap=200, n_cone=120, n_theta=240)
    exact = closed_form_CA(rn, th, rb)
    print(f"{name:<34}{c['C_D']:>12.6f}{exact:>13.6f}{abs(c['C_D']-exact)/exact:>10.2e}")

print()
print("=" * 78)
print("2. GRID CONVERGENCE  (Orion-like segment, Rn/Rb = 2.4)")
print("=" * 78)
th_orion = theta_c_for_spherical_segment(2.4, 1.0)
exact = closed_form_CA(2.4, th_orion, 1.0)
print(f"  closed form C_D = {exact:.8f}")
print(f"{'n_cap':>8}{'n_theta':>9}{'panels':>10}{'panel C_D':>14}{'rel err':>11}")
prev = None
for n in [10, 20, 40, 80, 160, 320]:
    c = coefficients(2.4, th_orion, 1.0, n_cap=n, n_cone=2, n_theta=4 * n)
    npan = n * 4 * n + 2 * 4 * n
    err = abs(c['C_D'] - exact) / exact
    rate = "" if prev is None else f"  (x{prev/err:.1f})"
    print(f"{n:>8}{4*n:>9}{npan:>10}{c['C_D']:>14.8f}{err:>11.2e}{rate}")
    prev = err
print("\n  Error falls ~4x per doubling: second-order, as expected for")
print("  midpoint integration on a smooth surface.")

print()
print("=" * 78)
print("3. THE THEORETICAL BOUND")
print("=" * 78)
print("  For ANY convex body in Newtonian flow, referenced to frontal area:")
print()
print("      C_A = (1/A_f) * int Cp (-n.x) dA   and   Cp <= Cp_max")
print("      int (-n.x) dA over the windward side = A_f exactly")
print("      =>  C_A <= Cp_max")
print()
print(f"  Cp_max = {CPMAX:.4f} for gamma = 1.4 (and only rises to ~1.95 for")
print("  a fully dissociated gas at gamma ~ 1.15).")
print()
print(f"  Baseline value in use:  C_D = 6.14")
print(f"  Newtonian ceiling:      C_D = {CPMAX:.4f}")
print(f"  ratio:                  {6.14/CPMAX:.2f}x over the ceiling")
print()
print("  So 6.14 cannot be a frontal-area drag coefficient for a convex body,")
print("  whatever the shape. No geometry fixes this.")

print()
print("=" * 78)
print("4. WHAT THE BASELINE GEOMETRY ACTUALLY GIVES")
print("=" * 78)
c = coefficients(2.4, th_orion, 1.0, n_cap=200, n_cone=120, n_theta=240)
print(f"  Orion-like spherical segment, R_n/R_b = 2.4")
print(f"    cone half-angle at tangency = {th_orion:.3f} deg")
print(f"    C_D (modified Newtonian)    = {c['C_D']:.4f}")
print(f"    C_L                         = {c['C_L']:.2e}   (zero by symmetry)")
print()
print(f"  Replacing 6.14 with {c['C_D']:.4f} changes the ballistic coefficient")
print(f"  beta = M/(C_D A) by a factor of {6.14/c['C_D']:.2f}.")

print()
print("=" * 78)
print("5. ANGLE OF ATTACK — where lift comes from")
print("=" * 78)
print(f"{'alpha [deg]':>12}{'C_D':>10}{'C_L':>10}{'L/D':>10}")
for a in [0, 5, 10, 15, 20, 25, 30]:
    c = coefficients(2.4, th_orion, 1.0, alpha_deg=a,
                     n_cap=120, n_cone=60, n_theta=180)
    print(f"{a:>12}{c['C_D']:>10.4f}{c['C_L']:>10.4f}{c['LoD']:>10.4f}")
print()
print("  The baseline fixes C_L = 0, which is exactly right at alpha = 0 for")
print("  an axisymmetric body. Lift requires a trimmed angle of attack, which")
print("  requires a center-of-gravity offset. That is a design variable the")
print("  current formulation does not have.")
