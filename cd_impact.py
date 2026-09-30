"""
What does the C_D = 6.14 question actually change?

Resolves where 6.14 sits relative to Newtonian theory, confirms where the
mass-model constant k = 300 kg/m^2 came from, and propagates the corrected
drag coefficient through the trajectory.
"""
import numpy as np
import reference_model as rm
from newtonian_ref import coefficients, closed_form_CA, cp_max

CPMAX = cp_max()
TH_ORION = np.degrees(np.arccos(1.0 / 2.4))
CD_NEWTON = closed_form_CA(2.4, TH_ORION, 1.0)

print("=" * 78)
print("1. WHERE THE PAPER'S NUMBERS COME FROM")
print("=" * 78)
print("  Di Fiore et al. (2021), Tables 1 and 3:")
print("     R (frontal section)  = 5.0 m")
print("     R_N                  = 2.4 R = 12.0 m")
print("     A_ref                = 78.54 m^2")
print("     M                    = 7500 kg")
print("     C_D                  = 6.14,  C_L = 0")
print()
print(f"  pi * 5.0^2 = {np.pi*25:.4f}, so A_ref = pi R^2 with R = 5.0 m used as")
print("  a RADIUS. The capsule is therefore 10 m across: a 2x-scaled Orion.")
print(f"  R_N/R_b = 12.0/5.0 = 2.40, and the real Orion is "
      f"6.035/2.515 = {6.035/2.515:.2f}. Same shape, twice the size.")
print()
print("  Kavin's mass constant k = 300 kg/m^2 follows directly:")
print(f"     paper   M / A_ref = 7500 / 78.54 = {7500/78.54:.4f} kg/m^2")
print(f"     model   M / A_ref = k R^2/(pi R^2) = k/pi")
print(f"     => k = pi * {7500/78.54:.4f} = {np.pi*7500/78.54:.2f}")
print("  So k = 300 was reverse-engineered to reproduce the paper's mass per")
print("  unit reference area. It is not arbitrary.")

print()
print("=" * 78)
print("2. C_D = 6.14 AGAINST THE NEWTONIAN CEILING")
print("=" * 78)
print("  For any convex body, referenced to its own frontal area A_f:")
print("      C_A = (1/A_f) * integral of Cp (-n.x) dA")
print("      integral of (-n.x) dA over the windward side = A_f  exactly")
print("      Cp <= Cp_max  everywhere")
print("      =>  C_A <= Cp_max,  with equality only for a flat plate")
print()
print(f"  Cp_max = {CPMAX:.4f}  (gamma = 1.4; about 1.95 for a dissociated")
print("  gas at gamma ~ 1.15, so the ceiling never approaches 6).")
print()
print(f"  C_D in use       : 6.14")
print(f"  Newtonian ceiling: {CPMAX:.4f}")
print(f"  over the ceiling : {6.14/CPMAX:.2f}x")
print()
print(f"  Panel model on the actual baseline shape (R_n/R_b = 2.4):")
print(f"      C_D = {CD_NEWTON:.4f}   referenced to pi R_b^2")
print(f"  6.14 / {CD_NEWTON:.4f} = {6.14/CD_NEWTON:.2f}")
print()
print("  For 6.14 to be self-consistent, A_ref would have to be")
print(f"  {CD_NEWTON/6.14:.4f} x the frontal area, i.e. an effective radius of")
print(f"  {np.sqrt(CD_NEWTON/6.14)*5.0:.3f} m rather than 5.0 m. No stated")
print("  dimension in the paper corresponds to that.")

print()
print("=" * 78)
print("3. BALLISTIC COEFFICIENT")
print("=" * 78)
beta_now = 300.0 / (6.14 * np.pi)
beta_fix = 300.0 / (CD_NEWTON * np.pi)
A_orion = np.pi * 2.515 ** 2
beta_real = 8900.0 / (1.35 * A_orion)
print(f"  baseline as coded        beta = k/(C_D pi) = {beta_now:>7.2f} kg/m^2")
print(f"  with Newtonian C_D                         = {beta_fix:>7.2f} kg/m^2")
print(f"  real Orion (8900 kg, C_D 1.35, A 19.9 m^2) = {beta_real:>7.2f} kg/m^2")
print()
print(f"  The correction raises beta by {beta_fix/beta_now:.2f}x, but the")
print(f"  baseline vehicle is still {beta_real/beta_fix:.0f}x lighter per unit")
print("  drag area than the capsule it is meant to represent. The mass model")
print("  is the remaining gap, and it is the next thing to look at.")

print()
print("=" * 78)
print("4. TRAJECTORY EFFECT OF THE CORRECTED C_D")
print("=" * 78)


def run(cd, gamma0=-0.1187, Fn=521.0, Fv=31050.8, R=5.9433):
    rm.C_D = cd
    return rm.simulate(gamma0=gamma0, Fn=Fn, Fv=Fv, R=R)


print("  At the optimized design point from the baseline reproduction:")
print(f"{'C_D':>10}{'beta':>10}{'h_final [km]':>15}{'V_final [m/s]':>15}"
      f"{'peak q':>13}")
rows = []
for cd in [6.14, CD_NEWTON]:
    s = run(cd)
    beta = s['M'] / (cd * s['A_ref'])
    rows.append((cd, beta, s))
    print(f"{cd:>10.3f}{beta:>10.2f}{s['h'][-1]/1000:>15.3f}"
          f"{s['V'][-1]:>15.2f}{s['q_coded'].max():>13.4e}")
rm.C_D = 6.14

a, b = rows[0][2], rows[1][2]
print()
print(f"  final altitude  {a['h'][-1]/1000:.2f} km -> {b['h'][-1]/1000:.2f} km")
print(f"  final velocity  {a['V'][-1]:.0f} m/s -> {b['V'][-1]:.0f} m/s   "
      f"({b['V'][-1]/a['V'][-1]:.1f}x)")
print(f"  peak heat flux  {a['q_coded'].max():.3e} -> {b['q_coded'].max():.3e}"
      f"   ({b['q_coded'].max()/a['q_coded'].max():.2f}x)")
print()
print("  The 25-50 km final-altitude constraint is active in the baseline, so")
print("  correcting C_D moves the optimum, not just the trajectory. This has")
print("  to be settled before a geometry-to-aero model replaces the constant,")
print("  or the change will look like a modeling regression.")
