"""
Forward vs reverse mode AD on the reentry model, measured rather than argued.

The cost model says forward scales with the number of inputs and reverse with
the number of outputs. This problem currently has 4 inputs and 102 outputs
(1 final altitude + 101 heat flux nodes), so forward should win now. The
question is where the crossover sits as design variables are added.

To sweep the input count without building real geometry models yet, extra
design variables are added as dummy inputs that perturb the radius by a
negligible amount. They add columns to the Jacobian without changing the
physics, which is exactly what is needed to measure AD cost scaling.
"""
import time, warnings
import numpy as np
import csdl_alpha as csdl

warnings.filterwarnings("ignore")
from reentry_model import build_trajectory

Q_MAX, H_MIN, H_MAX = 8e-5, 25000.0, 50000.0


def build(n_extra_dv=0, aggregate_constraints=False):
    rec = csdl.Recorder(inline=False)
    rec.start()

    gamma0 = csdl.Variable(name='gamma0', value=np.array([-0.1188]))
    Fv     = csdl.Variable(name='Fv',     value=np.array([1223.0]))
    Fn     = csdl.Variable(name='Fn',     value=np.array([1.0]))
    R      = csdl.Variable(name='R',      value=np.array([5.97]))
    for v, s in [(gamma0, 1.0), (Fv, 1.46e-5), (Fn, 1e-5), (R, 1/50)]:
        v.set_as_design_variable(scaler=s)

    # dummy design variables: real Jacobian columns, negligible physics
    R_eff = R
    for i in range(n_extra_dv):
        d = csdl.Variable(name=f'g{i}', value=np.array([0.0]))
        d.set_as_design_variable(scaler=1.0)
        R_eff = R_eff + 1e-9 * d

    states, q_dot, M, m_p = build_trajectory(gamma0, Fn, Fv, R_eff)

    h_final = states['h'][-1]
    h_final.set_as_constraint(lower=H_MIN, upper=H_MAX, scaler=1e-6)

    if aggregate_constraints:
        # one KS-aggregated constraint instead of 101 pointwise ones
        agg = csdl.maximum(q_dot, rho=1e8)
        agg.set_as_constraint(upper=Q_MAX, scaler=1e4)
        n_con = 2
    else:
        q_dot.set_as_constraint(upper=Q_MAX, scaler=1e4)
        n_con = 102

    obj = 1e-5 * M + 1e4 * csdl.maximum(q_dot, rho=1e8)
    obj.set_as_objective()
    rec.stop()
    return rec, 4 + n_extra_dv, n_con


def time_mode(rec, mode, repeats=3):
    """Compile, then time repeated derivative evaluations."""
    t0 = time.time()
    sim = csdl.experimental.JaxSimulator(
        recorder=rec, derivatives_kwargs={'loop': False, 'mode': mode})
    sim.run()
    sim.compute_optimization_derivatives()          # forces compilation
    t_compile = time.time() - t0

    t0 = time.time()
    for _ in range(repeats):
        sim.compute_optimization_derivatives()
    t_eval = (time.time() - t0) / repeats
    return t_compile, t_eval


if __name__ == '__main__':
    print("csdl_alpha 0.0.0a2: csdl.derivative(mode='forward') raises")
    print("NotImplementedError. Only reverse mode exists, so the mode choice")
    print("cannot be tested in CSDL today. What can be measured is how reverse")
    print("scales with the input and output counts.\n")

    for agg, label in [(False, "FULL CONSTRAINTS (102 outputs)"),
                       (True,  "AGGREGATED CONSTRAINTS (2 outputs)")]:
        print("=" * 68)
        print(f"REVERSE MODE SCALING -- {label}")
        print("=" * 68)
        print(f"{'n_inputs':>9} {'n_outputs':>10} {'compile':>11} {'eval':>12} "
              f"{'eval/input':>12}")
        for n_extra in [0, 12, 28, 60, 124]:
            rec, n_in, n_con = build(n_extra, aggregate_constraints=agg)
            c, e = time_mode(rec, 'reverse')
            print(f"{n_in:>9} {n_con:>10} {c:>10.2f}s {e*1000:>11.2f}ms "
                  f"{e*1000/n_in:>11.3f}ms")
        print()
