import numpy as np
 
G0 = 9.80665          # m/s^2
M_AIR = 0.0289644     # kg/mol
R_STAR = 8.31432      # J/(mol K)
R_E = 6356766.0       # m, radius used for the geopotential conversion
 
H_B = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0]
T_B = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
L_B = [-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3]
H_MAX = 84852.0       # geopotential m (86 km geometric)
 
 
def _pressure(P_b, T_b, L_b, T, dh):
    """Pressure a height dh above a layer base."""
    if L_b != 0.0:
        return P_b * (T_b / T) ** (G0 * M_AIR / (R_STAR * L_b))
    return P_b * np.exp(-G0 * M_AIR * dh / (R_STAR * T_b))
 
 
P_B = [101325.0]
for i in range(len(H_B) - 1):
    dh = H_B[i + 1] - H_B[i]
    P_B.append(_pressure(P_B[i], T_B[i], L_B[i], T_B[i] + L_B[i] * dh, dh))
 
 
def atmosphere_geopotential(h):
    """(rho, T, P) at geopotential altitude h [m]."""
    if h < 0.0 or h > H_MAX:
        raise ValueError(f"h = {h:.1f} m is outside 0 to {H_MAX:.0f} m")
    i = max(j for j in range(len(H_B)) if h >= H_B[j])
    T = T_B[i] + L_B[i] * (h - H_B[i])
    P = _pressure(P_B[i], T_B[i], L_B[i], T, h - H_B[i])
    return P * M_AIR / (R_STAR * T), T, P
 
 
def atmosphere(z):
    """(rho, T, P) at geometric altitude z [m]."""
    return atmosphere_geopotential(R_E * z / (R_E + z))
RE_EARTH = 6378000.0   # m
GRAV0 = 9.81           # m/s^2, sea-level gravity


def gravity(h):
    return GRAV0 * (RE_EARTH / (RE_EARTH + h)) ** 2


def eom(t, state, Fv, Fn, M, A_ref, C_D, C_L, rho_fn=None):
    V, gamma, h, beta = state
    rho = rho_fn(h) if rho_fn is not None else atmosphere(h)[0]
    g = gravity(h)
    q = 0.5 * rho * V ** 2 * A_ref
    D = q * C_D
    L = q * C_L
    dV = -(D + Fv) / M - g * np.sin(gamma)
    dgamma = ((L + Fn) / (M * V)
              - g * np.cos(gamma) / V
              + V / (h + RE_EARTH) * np.cos(gamma))
    dh = V * np.sin(gamma)
    dbeta = V * np.cos(gamma) / (h + RE_EARTH)
    return [dV, dgamma, dh, dbeta]
 
if __name__ == "__main__":
    table = [
        (0.0,     288.150, 101325.0,  1.225),
        (11000.0, 216.650, 22632.1,   0.363918),
        (20000.0, 216.650, 5474.89,   0.0880348),
        (32000.0, 228.650, 868.019,   0.013225),
        (47000.0, 270.650, 110.906,   0.00142753),
        (71000.0, 214.650, 3.95642,   6.4211e-05),
    ]
    ok = True
    for h, T_ref, P_ref, rho_ref in table:
        rho, T, P = atmosphere_geopotential(h)
        good = (abs(T - T_ref) < 5e-3 and abs(P / P_ref - 1) < 1e-5
                and abs(rho / rho_ref - 1) < 1e-4)
        ok &= good
        print(f"h={h:7.0f}  T={T:8.3f}  P={P:11.5g}  rho={rho:11.6g}  "
              f"{'PASS' if good else 'FAIL'}")
    print("geometric 86 km -> geopotential",
          f"{R_E * 86000.0 / (R_E + 86000.0):.1f} m")
    print("ALL PASS" if ok else "SOMETHING FAILED")

RE_EARTH = 6378000.0   # m
GRAV0 = 9.81           # m/s^2, sea-level gravity
 
 
def gravity(h):
    """Inverse-square gravity at geometric altitude h [m]."""
    return GRAV0 * (RE_EARTH / (RE_EARTH + h)) ** 2
 
 
def eom(t, state, Fv, Fn, M, A_ref, C_D, C_L, rho_fn=None):
    """
    Planar entry equations of motion over a spherical, non-rotating Earth.
 
    state = [V, gamma, h, beta]  (m/s, rad, m geometric, rad)
    Fv, Fn = thrust along / normal to the flight path [N] at this instant
             (the burn schedule is the caller's job; t is unused).
    gamma is negative while descending.
    rho_fn(h) -> density; defaults to the US76 model above, which raises above
    86 km until the 86-125 km question is settled.
    """
    V, gamma, h, beta = state
    rho = rho_fn(h) if rho_fn is not None else atmosphere(h)[0]
    g = gravity(h)
    q = 0.5 * rho * V ** 2 * A_ref
    D = q * C_D
    L = q * C_L
    dV = -(D + Fv) / M - g * np.sin(gamma)
    dgamma = ((L + Fn) / (M * V)
              - g * np.cos(gamma) / V
              + V / (h + RE_EARTH) * np.cos(gamma))
    dh = V * np.sin(gamma)
    dbeta = V * np.cos(gamma) / (h + RE_EARTH)
    return [dV, dgamma, dh, dbeta]
 
 
if __name__ == "__main__":
    # published values, geopotential altitude: h, T, P, rho
    table = [
        (0.0,     288.150, 101325.0,  1.225),
        (11000.0, 216.650, 22632.1,   0.363918),
        (20000.0, 216.650, 5474.89,   0.0880348),
        (32000.0, 228.650, 868.019,   0.013225),
        (47000.0, 270.650, 110.906,   0.00142753),
        (71000.0, 214.650, 3.95642,   6.4211e-05),
    ]
    ok = True
    for h, T_ref, P_ref, rho_ref in table:
        rho, T, P = atmosphere_geopotential(h)
        good = (abs(T - T_ref) < 5e-3 and abs(P / P_ref - 1) < 1e-5
                and abs(rho / rho_ref - 1) < 1e-4)
        ok &= good
        print(f"h={h:7.0f}  T={T:8.3f}  P={P:11.5g}  rho={rho:11.6g}  "
              f"{'PASS' if good else 'FAIL'}")
    print("geometric 86 km -> geopotential",
          f"{R_E * 86000.0 / (R_E + 86000.0):.1f} m")
    print("ALL PASS" if ok else "SOMETHING FAILED")
 
    # eom consistency check (evaluates the function once, no integration):
    # at circular speed, gamma = 0, no thrust, no lift, gamma_dot must be ~0
    h0 = 80000.0
    Vc = (gravity(h0) * (RE_EARTH + h0)) ** 0.5
    dV, dg, dh, db = eom(0.0, [Vc, 0.0, h0, 0.0], 0.0, 0.0, 7500.0, 78.54, 6.14, 0.0)
    print(f"circular check: gamma_dot={dg:.3e} (want ~0), h_dot={dh:.3e}, "
          f"beta_dot={db:.4e} (want {Vc / (RE_EARTH + h0):.4e}), V_dot={dV:.3f} (drag only, <0)")
 
