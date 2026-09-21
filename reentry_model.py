"""
Reentry trajectory model in CSDL alpha + Ozone.

Low-fidelity model from Di Fiore et al. (2021), Struct Multidisc Optim,
as used in kavin-nr/reentry-optimization.

States: V (velocity), gamma (flight path angle), h (altitude), B (downrange)
"""
import numpy as np
import csdl_alpha as csdl
import ozone

# ---------------------------------------------------------------- constants
G0    = 9.81        # sea-level gravity, m/s^2
RE    = 6378000.0   # Earth radius, m
C_EX  = 2305.0      # effective exhaust velocity, m/s
C_L   = 0.0         # lift coefficient (Oswatitsch Mach-independence limit)
C_D   = 6.14        # drag coefficient (same source)
K_TPS = 300.0       # rho_TPS * s_TPS, kg/m^2
KS    = 5.1564e-5   # heat flux constant, Sutton-Graves form
EPS   = 1e-6        # guards the thrust-magnitude norm at the origin

V0, H0, B0 = 7900.0, 125000.0, 0.0     # initial conditions
TF, N_STEPS, BURN_STEPS = 100.0, 100, 5


def atmosphere(h):
    """NASA Glenn simplified model, upper-stratosphere branch.
    h in m -> (rho kg/m^3, T Kelvin, p kPa)."""
    T_K = (-131.21 + 0.00299 * h) + 273.1
    p   = 2.488 * (T_K / 216.6) ** -11.388
    rho = p / (0.2869 * T_K)
    return rho, T_K, p


def heat_flux(rho, V, Rn):
    """Stagnation-point convective heating."""
    return KS * (rho / Rn) ** 0.5 * (V / 1000.0) ** 3.15


def compute_mass(Fn, Fv, burn_time, R):
    """Propellant mass from the rocket equation, plus TPS mass.

    EPS guards the Euclidean norm. d/dFn sqrt(Fn^2+Fv^2) = Fn/sqrt(Fn^2+Fv^2)
    is 0/0 at the origin, and SLSQP probes that corner because both thrust
    lower bounds are zero. EPS = 1e-6 N^2 shifts F by under a millinewton
    at any thrust the optimizer actually visits.
    """
    F   = (Fn**2 + Fv**2 + EPS) ** 0.5
    m_p = F * burn_time / C_EX
    return m_p + K_TPS * R**2, m_p


def ode_function(ov: ozone.ODEVars, M, R):
    """Planar equations of motion over a spherical, non-rotating Earth."""
    V, gam, h = ov.states['V'], ov.states['gamma'], ov.states['h']
    Fn = ov.dynamic_parameters['Fn']
    Fv = ov.dynamic_parameters['Fv']

    rho, _, _ = atmosphere(h)
    g     = G0 * (RE / (RE + h)) ** 2
    A_ref = np.pi * R**2
    Rn    = 2.4 * R

    q_dyn = 0.5 * rho * V**2 * A_ref
    L = q_dyn * C_L
    D = q_dyn * C_D

    ov.d_states['V']     = -((D + Fv) / M + g * csdl.sin(gam))
    ov.d_states['gamma'] = (1.0 / V) * ((L + Fn) / M - g * csdl.cos(gam)
                                        + (V**2 / (h + RE)) * csdl.cos(gam))
    ov.d_states['h']     = V * csdl.sin(gam)
    ov.d_states['B']     = V * csdl.cos(gam) / (h + RE)

    ov.profile_outputs['q_dot'] = heat_flux(rho, V, Rn)


def build_trajectory(gamma0, Fn, Fv, R, n_steps=N_STEPS, burn_steps=BURN_STEPS):
    """Assemble the ODE and integrate. Returns (states, q_dot, M, m_p)."""
    burn_time = TF / n_steps * burn_steps
    M, m_p = compute_mass(Fn, Fv, burn_time, R)

    # Thrust active for the first burn_steps nodes, zero afterwards.
    # Dynamic parameters need one value per time node (n_steps + 1).
    zeros = csdl.Variable(value=np.zeros(n_steps + 1 - burn_steps))
    Fn_t = csdl.concatenate([csdl.expand(Fn, (burn_steps,)), zeros])
    Fv_t = csdl.concatenate([csdl.expand(Fv, (burn_steps,)), zeros])

    ode = ozone.ODEProblem(ozone.methods.RK4(), ozone.approaches.TimeMarching())
    ode.add_state('V',     csdl.Variable(value=V0),  store_history=True)
    ode.add_state('gamma', gamma0,                   store_history=True)
    ode.add_state('h',     csdl.Variable(value=H0),  store_history=True)
    ode.add_state('B',     csdl.Variable(value=B0),  store_history=True)
    ode.add_dynamic_parameter('Fn', Fn_t)
    ode.add_dynamic_parameter('Fv', Fv_t)
    ode.set_timespan(ozone.timespans.StepVector(
        start=0.0, step_vector=np.full(n_steps, TF / n_steps)))
    ode.set_function(ode_function, M, R)

    out = ode.solve()
    return out.states, out.profile_outputs['q_dot'], M, m_p
