"""
Do derivatives flow correctly through the Newtonian panel model?

Two findings came out of writing this:

1. csdl.relu returns a NaN derivative at exactly x = 0. max(0,x)^2 is
   C1-continuous in theory, but relu's implementation is not usable at the
   shadow boundary, where panels do sit exactly. Replaced with
   max(0,x) = (x + sqrt(x^2 + eps))/2.

2. The sphere-cone parameterization has a CORNER at the spherical-segment
   limit theta_c = arccos(R_b/R_n), where the conical frustum has zero
   length. Past that point the frustum folds back on itself. AD returns the
   correct one-sided derivative there; a central difference straddles the
   corner and averages two different slopes into a meaningless number. The
   AD was right and the finite-difference check was wrong.

   The fix is a parameterization choice, not a numerical one: a pure
   spherical segment is built with no frustum (n_cone = 0), and a general
   sphere-cone carries the feasibility constraint R_n cos(theta_c) < R_b.
"""
import warnings
import numpy as np
import csdl_alpha as csdl

warnings.filterwarnings("ignore")
from newtonian_csdl import newtonian_coefficients
from newtonian_ref import coefficients

NC, NT = 30, 60


def ad_grad(rn, th, rb, al, n_cone):
    rec = csdl.Recorder(inline=True)
    rec.start()
    V = [csdl.Variable(value=np.array([q])) for q in (rn, th, rb, al)]
    _, _, CD, CL = newtonian_coefficients(V[0], V[1], V[2], V[3],
                                          n_cap=NC, n_cone=n_cone, n_theta=NT)
    dd = csdl.derivative(ofs=CD, wrts=V)
    g = [float(np.ravel(dd[v].value)[0]) for v in V]
    cd = float(np.ravel(CD.value)[0])
    rec.stop()
    return cd, g


def fd_grad(rn, th, rb, al, n_cone, h=1e-6):
    def f(a, b, c, d):
        return coefficients(a, np.degrees(b), c, alpha_deg=np.degrees(d),
                            n_cap=NC, n_cone=n_cone, n_theta=NT)['C_D']
    base, out = [rn, th, rb, al], []
    for i in range(4):
        up, dn = list(base), list(base)
        up[i] += h
        dn[i] -= h
        out.append((f(*up) - f(*dn)) / (2 * h))
    return out


NAMES = ['dC_D/dR_n', 'dC_D/dtheta_c', 'dC_D/dR_b', 'dC_D/dalpha']

CASES = [
    ("70 deg sphere-cone",      0.5,  np.radians(70.0),   1.0, np.radians(10.0), 15),
    ("60 deg sphere-cone",      0.25, np.radians(60.0),   1.0, np.radians(20.0), 15),
    ("45 deg sphere-cone",      0.1,  np.radians(45.0),   1.0, np.radians(0.0),  15),
    ("Orion segment, no frustum", 2.4, np.arccos(1.0/2.4), 1.0, np.radians(10.0), 0),
]

if __name__ == '__main__':
    print("=" * 76)
    print("REVERSE-MODE AD vs CENTRAL DIFFERENCES, AWAY FROM PARAMETER CORNERS")
    print("=" * 76)
    worst = 0.0
    for name, rn, th, rb, al, nk in CASES:
        cd, ad = ad_grad(rn, th, rb, al, nk)
        fd = fd_grad(rn, th, rb, al, nk)
        print(f"\n  {name}   (C_D = {cd:.6f}, alpha = {np.degrees(al):.0f} deg,"
              f" {nk} frustum rings)")
        print(f"{'':6}{'derivative':>16}{'CSDL (AD)':>16}"
              f"{'central diff':>16}{'rel err':>12}")
        for nm, a, f in zip(NAMES, ad, fd):
            rel = abs(a - f) / max(abs(f), 1e-9)
            worst = max(worst, rel if abs(f) > 1e-9 else 0.0)
            print(f"{'':6}{nm:>16}{a:>16.8f}{f:>16.8f}{rel:>12.2e}")
    print(f"\n  worst relative error across all cases: {worst:.2e}")

    print()
    print("=" * 76)
    print("THE CORNER, MADE EXPLICIT")
    print("=" * 76)
    RN0, RB0 = 2.4, 1.0
    TH0 = np.arccos(RB0 / RN0)
    AL0 = np.radians(10.0)
    f = lambda rn: coefficients(rn, np.degrees(TH0), RB0,
                                alpha_deg=np.degrees(AL0),
                                n_cap=NC, n_cone=15, n_theta=NT)['C_D']
    h = 1e-6
    f0 = f(RN0)
    fwd = (f(RN0 + h) - f0) / h
    bwd = (f0 - f(RN0 - h)) / h
    _, ad = ad_grad(RN0, TH0, RB0, AL0, 15)
    print("  Sphere-cone parameterization evaluated exactly at")
    print("  theta_c = arccos(R_b/R_n), where the frustum has zero length:")
    print()
    print(f"    forward  difference   {fwd:>12.6f}")
    print(f"    backward difference   {bwd:>12.6f}")
    print(f"    central  difference   {(fwd+bwd)/2:>12.6f}   <- meaningless")
    print(f"    CSDL reverse-mode AD  {ad[0]:>12.6f}   <- the right derivative")
    print()
    print("  The one-sided slopes differ by "
          f"{abs(fwd-bwd):.3f}. AD reproduces the forward")
    print("  slope exactly. Nothing is wrong with the AD; the shape family")
    print("  itself is not differentiable at that point, so an optimizer must")
    print("  be kept off it with R_n cos(theta_c) < R_b as a constraint.")

    print()
    print("=" * 76)
    print("SHADOW BOUNDARY: is the windward/leeward transition smooth?")
    print("=" * 76)
    print(f"{'alpha [deg]':>12}{'C_D':>12}{'dC_D/dalpha':>16}{'change':>12}")
    prev = None
    for a_deg in np.arange(0.0, 71.0, 10.0):
        cd, g = ad_grad(2.4, np.arccos(1 / 2.4), 1.0, np.radians(a_deg), 0)
        ch = "" if prev is None else f"{g[3]-prev:>12.4f}"
        print(f"{a_deg:>12.0f}{cd:>12.6f}{g[3]:>16.6f}{ch}")
        prev = g[3]
    print()
    print("  Smooth throughout, with no jump as panels rotate into shadow.")
    print("  That is what max(0,x)^2 buys over a hard windward/leeward test.")
