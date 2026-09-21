"""
One-step vs five-step burn, in the CSDL framework.

Changing burn_steps from 5 to 1 does two things at once:
  (A) cuts the delivered impulse by 5x      -> trajectory effect
  (B) cuts m_p = F*dt/c by 5x               -> propellant mass effect

To attribute the change, the burn duration used for the dynamics is
decoupled from the burn duration charged to the mass model. That split is
physically artificial; it is what makes the attribution possible.

Reference: Di Fiore et al. (2021) apply thrust on the first integration
step only. Kavin's implementation uses five.
"""
import warnings
import numpy as np
import csdl_alpha as csdl
import modopt as mo

warnings.filterwarnings("ignore")
import reentry_model as rm
from reentry_model import (atmosphere, heat_flux, C_EX, K_TPS, EPS,
                           G0, RE, C_L, C_D, V0, H0, B0, TF, N_STEPS)
import ozone


def build_split(gamma0, Fn, Fv, R, dyn_steps, mass_steps, n_steps=N_STEPS):
    """dyn_steps  -- number of nodes with thrust active in the EOM
       mass_steps -- number of nodes charged to the propellant mass model"""
    dt = TF / n_steps
    F   = (Fn**2 + Fv**2 + EPS) ** 0.5
    m_p = F * (mass_steps * dt) / C_EX
    M   = m_p + K_TPS * R**2

    zeros = csdl.Variable(value=np.zeros(n_steps + 1 - dyn_steps))
    Fn_t = csdl.concatenate([csdl.expand(Fn, (dyn_steps,)), zeros])
    Fv_t = csdl.concatenate([csdl.expand(Fv, (dyn_steps,)), zeros])

    ode = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.TimeMarching())
    ode.add_state('V',     csdl.Variable(value=V0), store_history=True)
    ode.add_state('gamma', gamma0,                  store_history=True)
    ode.add_state('h',     csdl.Variable(value=H0), store_history=True)
    ode.add_state('B',     csdl.Variable(value=B0), store_history=True)
    ode.add_dynamic_parameter('Fn', Fn_t)
    ode.add_dynamic_parameter('Fv', Fv_t)
    ode.set_timespan(ozone.timespans.StepVector(
        start=0.0, step_vector=np.full(n_steps, dt)))
    ode.set_function(rm.ode_function, M, R)
    out = ode.solve()
    return out.states, out.profile_outputs['q_dot'], M, m_p, F


def evaluate(gamma0, Fn, Fv, R, dyn_steps, mass_steps):
    """Single forward evaluation at a fixed design point."""
    rec = csdl.Recorder(inline=True)
    rec.start()
    g = csdl.Variable(value=np.array([gamma0]))
    fn = csdl.Variable(value=np.array([Fn]))
    fv = csdl.Variable(value=np.array([Fv]))
    r = csdl.Variable(value=np.array([R]))
    st, q, M, m_p, F = build_split(g, fn, fv, r, dyn_steps, mass_steps)
    qv = q.value.flatten()
    out = dict(M=float(M.value[0]), m_p=float(m_p.value[0]),
               h_f=float(st['h'].value.flatten()[-1]),
               V_f=float(st['V'].value.flatten()[-1]),
               q_peak=float(qv.max()),
               impulse=float(F.value[0]) * dyn_steps * (TF / N_STEPS))
    out['obj'] = 1e-5 * out['M'] + 1e4 * out['q_peak']
    rec.stop()
    return out


def optimize(dyn_steps, mass_steps, label, ftol=1e-6):
    """Re-optimize with the given burn model."""
    rec = csdl.Recorder(inline=False)
    rec.start()
    g0 = csdl.Variable(name='gamma0', value=np.array([-0.9]))
    Fv = csdl.Variable(name='Fv',     value=np.array([25000.0]))
    Fn = csdl.Variable(name='Fn',     value=np.array([25000.0]))
    R  = csdl.Variable(name='R',      value=np.array([12.0]))
    g0.set_as_design_variable(lower=-np.pi/2, upper=0.0,    scaler=1.0)
    Fv.set_as_design_variable(lower=0.0, upper=146000.,     scaler=1.46e-5)
    Fn.set_as_design_variable(lower=0.0, upper=100000.,     scaler=1e-5)
    R.set_as_design_variable(lower=5.0, upper=50.0,         scaler=1/50)

    st, q, M, m_p, F = build_split(g0, Fn, Fv, R, dyn_steps, mass_steps)
    hf = st['h'][-1]
    hf.set_as_constraint(lower=25000., upper=50000., scaler=1e-6)
    q.set_as_constraint(upper=8e-5, scaler=1e4)
    qp = csdl.maximum(q, rho=1e8)
    obj = 1e-5 * M + 1e4 * qp
    obj.set_as_objective()
    rec.stop()

    sim = csdl.experimental.JaxSimulator(
        recorder=rec, derivatives_kwargs={'loop': False},
        additional_outputs=[M, m_p, qp, hf, F])
    prob = mo.CSDLAlphaProblem(problem_name='burn', simulator=sim)
    opt = mo.SLSQP(prob, turn_off_outputs=True, solver_options={'ftol': ftol})
    opt.solve()
    sim.run()
    g = lambda v: float(np.ravel(sim[v])[0])
    imp = g(F) * dyn_steps * (TF / N_STEPS)
    res = dict(label=label, gamma0=g(g0), Fv=g(Fv), Fn=g(Fn), R=g(R),
               M=g(M), m_p=g(m_p), q_peak=g(qp), h_f=g(hf),
               obj=g(obj), impulse=imp, dV=imp / g(M),
               nfev=opt.results.get('nfev'))
    return res


BASE = dict(gamma0=-0.9, Fn=25000.0, Fv=25000.0, R=12.0)

if __name__ == '__main__':
    print("=" * 84)
    print("2x2 FACTORIAL at the initial guess  (CSDL / Ozone)")
    print("=" * 84)
    cases = [("baseline: 5-step dyn, 5-step mass", 5, 5),
             ("naive change: 1-step dyn, 1-step mass", 1, 1),
             ("trajectory effect only: 1-step dyn, 5-step mass", 1, 5),
             ("mass effect only: 5-step dyn, 1-step mass", 5, 1)]
    res = {}
    print(f"{'case':<50}{'M [kg]':>11}{'m_p':>8}{'h_f [km]':>10}"
          f"{'peak q':>12}{'obj':>9}")
    for label, d, m in cases:
        r = evaluate(**BASE, dyn_steps=d, mass_steps=m)
        res[label] = r
        print(f"{label:<50}{r['M']:>11.1f}{r['m_p']:>8.1f}"
              f"{r['h_f']/1000:>10.3f}{r['q_peak']:>12.4e}{r['obj']:>9.4f}")

    b  = res[cases[0][0]]; n = res[cases[1][0]]
    tj = res[cases[2][0]]; ms = res[cases[3][0]]
    print()
    print("ATTRIBUTION (change in objective from baseline)")
    print(f"  total change, naive 1-step      : {n['obj']-b['obj']:+.5f}")
    print(f"  trajectory effect alone         : {tj['obj']-b['obj']:+.5f}")
    print(f"  propellant mass effect alone    : {ms['obj']-b['obj']:+.5f}")
    s = (tj['obj']-b['obj']) + (ms['obj']-b['obj'])
    print(f"  sum of the two isolated effects : {s:+.5f}")
    print(f"  interaction (total minus sum)   : {(n['obj']-b['obj'])-s:+.5f}")
    print()
    print(f"  m_p is {100*b['m_p']/b['M']:.2f}% of total mass at 5 steps, "
          f"{100*n['m_p']/n['M']:.2f}% at 1 step")

    print()
    print("=" * 84)
    print("RE-OPTIMIZED UNDER EACH BURN MODEL")
    print("=" * 84)
    o5 = optimize(5, 5, "5-step (Kavin)")
    o1 = optimize(1, 1, "1-step (paper)")
    print(f"{'quantity':<18}{'5-step':>15}{'1-step':>15}{'change':>12}")
    for name, key in [('gamma0 [deg]', None), ('Fv [N]', 'Fv'), ('Fn [N]', 'Fn'),
                      ('R [m]', 'R'), ('M [kg]', 'M'), ('m_p [kg]', 'm_p'),
                      ('impulse [N.s]', 'impulse'), ('delta-V [m/s]', 'dV'),
                      ('peak q', 'q_peak'), ('objective', 'obj')]:
        if key is None:
            a, c = np.degrees(o5['gamma0']), np.degrees(o1['gamma0'])
        else:
            a, c = o5[key], o1[key]
        pct = 100*(c-a)/a if a else float('nan')
        print(f"{name:<18}{a:>15.4f}{c:>15.4f}{pct:>11.2f}%")
    print()
    print(f"  Entry velocity is {V0:.0f} m/s, so the burn is a "
          f"{100*o5['dV']/V0:.3f}% perturbation at 5 steps and "
          f"{100*o1['dV']/V0:.3f}% at 1 step.")
    print(f"  SLSQP evaluations: {o5['nfev']} (5-step), {o1['nfev']} (1-step)")
