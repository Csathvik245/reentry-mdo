"""
Regression test: CSDL/Ozone implementation vs the standalone NumPy model.

Run this after any change to reentry_model.py. The zero-thrust case must match
to machine precision. The thrust case will differ slightly because Ozone
interpolates dynamic parameters across RK4 stages while the NumPy model holds
them constant within a step.
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csdl_alpha as csdl
from reentry_model import atmosphere, compute_mass, heat_flux, build_trajectory
from reference_model import simulate

GOLD = {
    'rho_125km': 8.630751352088e-07,
    'T_125km':   515.640000000,
    'M':         43276.692709456,
    'q_ic':      6.000673482010e-06,
}

def close(a, b, tol=1e-9):
    return abs(a - b) / max(abs(b), 1e-30) < tol

rec = csdl.Recorder(inline=True)
rec.start()

print("COMPONENT CHECKS")
h = csdl.Variable(value=np.array([125000.0]))
rho, T, _ = atmosphere(h)
for label, got, want in [
        ('atmosphere rho', float(rho.value[0]), GOLD['rho_125km']),
        ('atmosphere T',   float(T.value[0]),   GOLD['T_125km'])]:
    print(f"  {'PASS' if close(got, want) else 'FAIL'}  {label:18s} "
          f"{got:.12e}  vs  {want:.12e}")

M, _ = compute_mass(csdl.Variable(value=np.array([25000.0])),
                    csdl.Variable(value=np.array([25000.0])), 5.0,
                    csdl.Variable(value=np.array([12.0])))
print(f"  {'PASS' if close(float(M.value[0]), GOLD['M'], 1e-8) else 'FAIL'}  "
      f"{'mass':18s} {float(M.value[0]):.9f}  vs  {GOLD['M']:.9f}")

q = heat_flux(rho, csdl.Variable(value=np.array([7900.0])), 28.8)
print(f"  {'PASS' if close(float(q.value[0]), GOLD['q_ic']) else 'FAIL'}  "
      f"{'heat flux':18s} {float(q.value[0]):.12e}  vs  {GOLD['q_ic']:.12e}")

print("\nZERO-THRUST TRAJECTORY  (must match to machine precision)")
z = csdl.Variable(value=np.array([0.0]))
st, _, _, _ = build_trajectory(csdl.Variable(value=np.array([-0.9])), z, z,
                               csdl.Variable(value=np.array([12.0])))
h_c = st['h'].value.flatten()
V_c = st['V'].value.flatten()
rec.stop()

ref = simulate(gamma0=-0.9, Fn=0.0, Fv=0.0, R=12.0)
# Relative tolerance. The EPS guard in compute_mass adds ~2e-6 kg of phantom
# propellant, which perturbs the drag deceleration at around the 12th digit.
dh = np.abs(h_c - ref['h']).max() / np.abs(ref['h']).max()
dV = np.abs(V_c - ref['V']).max() / np.abs(ref['V']).max()
print(f"  {'PASS' if dh < 1e-10 else 'FAIL'}  max relative altitude error  {dh:.3e}")
print(f"  {'PASS' if dV < 1e-10 else 'FAIL'}  max relative velocity error  {dV:.3e}")
