"""
Independent NumPy reference implementation of the reentry model.
No CSDL, no Ozone, no JAX. Ground truth for validating the CSDL version.

Equations from Di Fiore et al. (2021), Struct Multidisc Optim 64:3017-3035.
"""
import numpy as np

G0   = 9.81        # m/s^2
RE   = 6378000.0   # m
C_EX = 2305.0      # effective exhaust velocity, m/s
C_L  = 0.0         # lift coefficient (Oswatitsch Mach-independence limit)
C_D  = 6.14        # drag coefficient (same source)
K_AREAL = 300.0    # rho_TPS * s_TPS, kg/m^2
KS   = 5.1564e-5   # heat flux constant


def atmosphere_kavin(h):
    """NASA Glenn simplified model, upper-stratosphere branch only."""
    T_C = -131.21 + 0.00299 * h
    T_K = T_C + 273.1
    p   = 2.488 * (T_K / 216.6) ** -11.388
    rho = p / (0.2869 * T_K)
    return rho, T_K, p


def q_dot_as_coded(rho, V, Rn):
    return KS * np.sqrt(rho / Rn) * (V / 1000.0) ** 3.15


def compute_mass(Fn, Fv, burn_time, R):
    F   = np.sqrt(Fn**2 + Fv**2)
    m_p = F * burn_time / C_EX
    m   = K_AREAL * R**2
    return m_p + m, m_p, m


def derivatives(state, Fn, Fv, M, R, Rn, A_ref):
    """state = [V, gamma, h, B]"""
    V, gamma, h, B = state
    rho, _, _ = atmosphere_kavin(h)
    g = G0 * (RE / (RE + h)) ** 2
    L = 0.5 * rho * V**2 * A_ref * C_L
    D = 0.5 * rho * V**2 * A_ref * C_D

    dV     = -((D + Fv) / M + g * np.sin(gamma))
    dgamma = (1.0 / V) * ((L + Fn) / M - g * np.cos(gamma)
                          + (V**2 / (h + RE)) * np.cos(gamma))
    dh     = V * np.sin(gamma)
    dB     = V * np.cos(gamma) / (h + RE)
    return np.array([dV, dgamma, dh, dB])


def simulate(gamma0=-0.9, Fn=25000.0, Fv=25000.0, R=12.0,
             V0=7900.0, h0=125000.0, B0=0.0,
             tf=100.0, n_steps=100, burn_time=5.0):
    dt = tf / n_steps
    burn_steps = int(burn_time / dt)

    M, m_p, m_struct = compute_mass(Fn, Fv, burn_time, R)
    Rn    = 2.4 * R
    A_ref = np.pi * R**2

    Fn_arr = np.concatenate([np.full(burn_steps, Fn),
                             np.zeros(n_steps - burn_steps + 1)])
    Fv_arr = np.concatenate([np.full(burn_steps, Fv),
                             np.zeros(n_steps - burn_steps + 1)])

    state = np.array([V0, gamma0, h0, B0], dtype=float)
    hist  = [state.copy()]
    t     = [0.0]

    for i in range(n_steps):
        fn, fv = Fn_arr[i], Fv_arr[i]
        f = lambda s: derivatives(s, fn, fv, M, R, Rn, A_ref)
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        hist.append(state.copy())
        t.append(t[-1] + dt)

    hist = np.array(hist)
    t    = np.array(t)
    V, gamma, h, B = hist[:, 0], hist[:, 1], hist[:, 2], hist[:, 3]
    rho, T_K, p = atmosphere_kavin(h)

    return {
        "t": t, "V": V, "gamma": gamma, "h": h, "B": B,
        "rho": rho, "q_coded": q_dot_as_coded(rho, V, Rn),
        "M": M, "m_p": m_p, "m_struct": m_struct,
        "R": R, "Rn": Rn, "A_ref": A_ref,
    }
