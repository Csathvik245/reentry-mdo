"""
Where do the evaluation-count and timing differences against Kavin's
reported numbers come from?

Reported: SLSQP 6 evaluations / 25 callbacks / ~235 s
          COBYQA 341 evaluations / 5318 callbacks / ~272 s

Two hypotheses tested here:
  (1) convergence tolerance -- a looser ftol stops sooner
  (2) derivatives_kwargs loop=False unrolls 100 RK4 steps into the graph,
      which should dominate compile time
"""
import time, warnings
import numpy as np
import csdl_alpha as csdl
import modopt as mo

warnings.filterwarnings("ignore")
from run_optimization import build_problem


def solve_at(ftol, loop=False):
    rec, v = build_problem()
    t0 = time.time()
    sim = csdl.experimental.JaxSimulator(
        recorder=rec, derivatives_kwargs={'loop': loop},
        additional_outputs=[v['M'], v['q_peak'], v['h_final']])
    prob = mo.CSDLAlphaProblem(problem_name='reentry', simulator=sim)
    opt = mo.SLSQP(prob, turn_off_outputs=True, solver_options={'ftol': ftol})
    t_compile = time.time() - t0

    t0 = time.time()
    opt.solve()
    t_solve = time.time() - t0
    r = opt.results
    sim.run()
    return dict(ftol=ftol, loop=loop, t_compile=t_compile, t_solve=t_solve,
                nfev=r.get('nfev'), njev=r.get('njev'), nit=r.get('nit'),
                obj=float(np.ravel(sim[v['obj']])[0]),
                M=float(np.ravel(sim[v['M']])[0]))


print("=" * 78)
print("HYPOTHESIS 1 -- convergence tolerance drives the evaluation count")
print("=" * 78)
print(f"{'ftol':>10} {'nfev':>6} {'njev':>6} {'nit':>6} {'objective':>12} "
      f"{'mass [kg]':>12} {'solve [s]':>10}")
rows = []
for ftol in [1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
    r = solve_at(ftol)
    rows.append(r)
    print(f"{ftol:>10.0e} {r['nfev']:>6} {r['njev']:>6} {r['nit']:>6} "
          f"{r['obj']:>12.6f} {r['M']:>12.2f} {r['t_solve']:>10.2f}")

print()
print("Kavin reported 6 evaluations and 25 callbacks. The callback count")
print("matches njev closely at tight tolerance; the evaluation count only")
print("comes down to single digits at a much looser ftol.")

print()
print("=" * 78)
print("HYPOTHESIS 2 -- loop=False unrolls the RK4 loop and dominates compile")
print("=" * 78)
print(f"{'loop':>8} {'compile [s]':>13} {'solve [s]':>11} {'total [s]':>11} "
      f"{'nfev':>6} {'objective':>12}")
for loop in [False, True]:
    try:
        r = solve_at(1e-6, loop=loop)
        print(f"{str(loop):>8} {r['t_compile']:>12.2f}s {r['t_solve']:>10.2f}s "
              f"{r['t_compile']+r['t_solve']:>10.2f}s {r['nfev']:>6} "
              f"{r['obj']:>12.6f}")
    except Exception as e:
        print(f"{str(loop):>8}  failed: {type(e).__name__}: {e}")

print()
print("Reported total was ~235 s. If compile time here is a fraction of that,")
print("the wall-clock comparison is dominated by compilation rather than by")
print("the optimization itself, and the meaningful result is evaluation count.")
