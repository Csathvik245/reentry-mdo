"""
Run this before touching Kavin's code. It checks each layer in order.
The first FAIL tells you which layer is broken.

    python verify_env.py
"""
import sys, traceback
import numpy as np

def check(label, fn):
    try:
        result = fn()
        print(f"  PASS  {label}" + (f"   [{result}]" if result else ""))
        return True
    except Exception as e:
        print(f"  FAIL  {label}")
        print(f"        {type(e).__name__}: {e}")
        return False

print(f"\npython {sys.version.split()[0]}\n")

# --- Layer 1: imports -------------------------------------------------
print("Layer 1 -- packages import")
ok = True
ok &= check("numpy",      lambda: np.__version__)
ok &= check("jax",        lambda: __import__("jax").__version__)
ok &= check("csdl_alpha", lambda: __import__("csdl_alpha").__version__ if hasattr(__import__("csdl_alpha"), "__version__") else "installed")
ok &= check("modopt",     lambda: "installed" if __import__("modopt") else "")
ok &= check("ozone is the LSDO one, not the AWS package",
            lambda: "ok" if hasattr(__import__("ozone"), "ODEProblem") else (_ for _ in ()).throw(
                ImportError("wrong 'ozone' -- uninstall it and install from github.com/LSDOlab/ozone")))
if not ok:
    print("\nStop here and fix the imports before continuing.\n"); sys.exit(1)

import csdl_alpha as csdl

# --- Layer 2: graph construction --------------------------------------
print("\nLayer 2 -- Recorder builds a graph")
rec = csdl.Recorder(inline=False)
rec.start()
R  = csdl.Variable(value=np.array([12.0]), name="R")
g0 = csdl.Variable(value=np.array([-0.9]), name="g0")
R.set_as_design_variable()
g0.set_as_design_variable()
f = 300.0 * R**2 + 1000.0 * csdl.sin(g0)
f.name = "f"
f.set_as_objective()
rec.stop()
print(f"  PASS  graph built with {len(rec.active_graph.node_table)} nodes")

# --- Layer 3: JAX compiles and evaluates ------------------------------
print("\nLayer 3 -- JaxSimulator compiles and runs")
sim = csdl.experimental.JaxSimulator(recorder=rec)
sim.run()
val = float(np.ravel(sim[f])[0])
expected = 300.0 * 12.0**2 + 1000.0 * np.sin(-0.9)
assert abs(val - expected) < 1e-8, f"value mismatch: {val} vs {expected}"
print(f"  PASS  f = {val:.6f}  (hand: {expected:.6f})")

# --- Layer 4: derivatives are exact -----------------------------------
print("\nLayer 4 -- derivatives match hand calculation")
raw = sim.compute_optimization_derivatives()
grad = np.ravel(raw["df"] if isinstance(raw, dict) else raw)
hand = np.array([600.0 * 12.0, 1000.0 * np.cos(-0.9)])
for name, got, want in zip(["df/dR", "df/dg0"], grad, hand):
    err = abs(got - want)
    status = "PASS" if err < 1e-9 else "FAIL"
    print(f"  {status}  {name:8s} = {got:.10f}   hand = {want:.10f}   err = {err:.2e}")

# --- Layer 5: finite differences for contrast -------------------------
print("\nLayer 5 -- what finite differences would have given")
def f_np(Rv, gv):
    return 300.0 * Rv**2 + 1000.0 * np.sin(gv)
for h in [1e-2, 1e-6, 1e-10, 1e-14]:
    fd = (f_np(12.0 + h, -0.9) - f_np(12.0, -0.9)) / h
    print(f"  h = {h:.0e}   df/dR ~ {fd:18.10f}   err = {abs(fd - 7200.0):.3e}")
print("\n  Error falls, bottoms out near h=1e-8, then explodes as subtractive\n  cancellation takes over. AD has no h to tune.\n")
