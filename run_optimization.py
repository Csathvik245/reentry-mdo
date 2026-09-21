"""
Optimization driver: reproduce the SLSQP baseline, compare against COBYQA.

    python run_optimization.py slsqp
    python run_optimization.py cobyqa
    python run_optimization.py both

Design variables : gamma0, Fv, Fn, R
Objective        : 1e-5 * M + 1e4 * max(q_dot)
Constraints      : 25 km <= h_final <= 50 km,  q_dot <= 8e-5 at every node
"""
import sys, time, warnings
import numpy as np
import csdl_alpha as csdl
import modopt as mo

warnings.filterwarnings("ignore")
from reentry_model import build_trajectory

SC_GAMMA, SC_FV, SC_FN, SC_R = 1.0, 1.46e-5, 1e-5, 1.0 / 50.0
Q_MAX        = 8e-5
H_MIN, H_MAX = 25000.0, 50000.0
BURN_STEPS   = 5


def build_problem(burn_steps=BURN_STEPS, ftol=1e-6):
    rec = csdl.Recorder(inline=False)
    rec.start()

    gamma0 = csdl.Variable(name='gamma0', value=np.array([-0.9]))
    Fv     = csdl.Variable(name='Fv',     value=np.array([25000.0]))
    Fn     = csdl.Variable(name='Fn',     value=np.array([25000.0]))
    R      = csdl.Variable(name='R',      value=np.array([12.0]))

    gamma0.set_as_design_variable(lower=-np.pi/2, upper=0.0,     scaler=SC_GAMMA)
    Fv.set_as_design_variable(    lower=0.0,      upper=146000., scaler=SC_FV)
    Fn.set_as_design_variable(    lower=0.0,      upper=100000., scaler=SC_FN)
    R.set_as_design_variable(     lower=5.0,      upper=50.0,    scaler=SC_R)

    states, q_dot, M, m_p = build_trajectory(gamma0, Fn, Fv, R,
                                             burn_steps=burn_steps)

    h_final = states['h'][-1]
    h_final.name = 'h_final'
    h_final.set_as_constraint(lower=H_MIN, upper=H_MAX, scaler=1e-6)

    q_dot.name = 'q_dot'
    q_dot.set_as_constraint(upper=Q_MAX, scaler=1e4)

    q_peak = csdl.maximum(q_dot, rho=1e8)
    obj = 1e-5 * M + 1e4 * q_peak
    obj.name = 'objective'
    obj.set_as_objective(scaler=1.0)

    M.name, m_p.name, q_peak.name = 'M', 'm_p', 'q_peak'
    rec.stop()
    v = dict(gamma0=gamma0, Fv=Fv, Fn=Fn, R=R,
             M=M, m_p=m_p, q_peak=q_peak, h_final=h_final, obj=obj)
    return rec, v


def run(optimizer_name, burn_steps=BURN_STEPS, ftol=1e-6, verbose=True):
    rec, v = build_problem(burn_steps)

    t0 = time.time()
    sim = csdl.experimental.JaxSimulator(
        recorder=rec, derivatives_kwargs={'loop': False},
        additional_outputs=[v['M'], v['m_p'], v['q_peak'], v['h_final']])
    prob = mo.CSDLAlphaProblem(problem_name='reentry', simulator=sim)
    t_compile = time.time() - t0

    if optimizer_name == 'slsqp':
        opt = mo.SLSQP(prob, turn_off_outputs=True, solver_options={'ftol': ftol})
    else:
        opt = mo.COBYQA(prob, turn_off_outputs=True)

    t0 = time.time()
    opt.solve()
    t_solve = time.time() - t0

    res = opt.results
    sim.run()

    out = dict(
        name=optimizer_name, ftol=ftol,
        t_compile=t_compile, t_solve=t_solve, t_total=t_compile + t_solve,
        nfev=res.get('nfev', None), njev=res.get('njev', None),
        nit=res.get('nit', None),
        objective=float(np.ravel(sim[v['obj']])[0]),
        M=float(np.ravel(sim[v['M']])[0]),
        m_p=float(np.ravel(sim[v['m_p']])[0]),
        q_peak=float(np.ravel(sim[v['q_peak']])[0]),
        h_final=float(np.ravel(sim[v['h_final']])[0]),
        gamma0=float(np.ravel(sim[v['gamma0']])[0]),
        Fv=float(np.ravel(sim[v['Fv']])[0]),
        Fn=float(np.ravel(sim[v['Fn']])[0]),
        R=float(np.ravel(sim[v['R']])[0]),
    )
    if verbose:
        report(out)
    return out


def report(o):
    print(f"\n{'='*62}\n{o['name'].upper()}  (ftol={o['ftol']:.0e})\n{'='*62}")
    print(f"  compile time        {o['t_compile']:10.2f} s")
    print(f"  solve time          {o['t_solve']:10.2f} s")
    print(f"  total               {o['t_total']:10.2f} s")
    print(f"  function evals      {o['nfev']}")
    print(f"  gradient evals      {o['njev']}")
    print(f"  iterations          {o['nit']}")
    print(f"  ---")
    print(f"  gamma0    {o['gamma0']:12.6f} rad  ({np.degrees(o['gamma0']):.3f} deg)")
    print(f"  Fv        {o['Fv']:12.2f} N")
    print(f"  Fn        {o['Fn']:12.2f} N")
    print(f"  R         {o['R']:12.5f} m")
    print(f"  ---")
    print(f"  mass      {o['M']:12.3f} kg   (propellant {o['m_p']:.2f})")
    print(f"  peak q    {o['q_peak']:12.6e}   (limit {Q_MAX:.1e})")
    print(f"  h_final   {o['h_final']/1000:12.4f} km  (band 25 to 50)")
    print(f"  objective {o['objective']:12.6f}")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'slsqp'
    if which == 'both':
        a = run('slsqp'); b = run('cobyqa')
        print(f"\n{'='*62}\nCOMPARISON\n{'='*62}")
        print(f"{'':<20}{'SLSQP':>16}{'COBYQA':>16}")
        for label, key in [('function evals', 'nfev'), ('gradient evals', 'njev'),
                           ('solve time [s]', 't_solve'), ('total time [s]', 't_total'),
                           ('objective', 'objective'), ('mass [kg]', 'M')]:
            x, y = a[key], b[key]
            fx = f"{x:16.4f}" if isinstance(x, float) else f"{str(x):>16}"
            fy = f"{y:16.4f}" if isinstance(y, float) else f"{str(y):>16}"
            print(f"{label:<20}{fx}{fy}")
        if a['nfev'] and b['nfev']:
            print(f"\nevaluation count ratio: {b['nfev']/a['nfev']:.1f}x")
    else:
        run(which)
