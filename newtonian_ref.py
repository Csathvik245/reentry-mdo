"""
Modified Newtonian aerodynamics for axisymmetric blunt bodies — NumPy reference.

Cp = Cp_max * max(0, -n.u)^2      (Lees 1955)

Geometry is a spherically blunted cone: nose radius R_n, cone half-angle
theta_c measured from the axis, base radius R_b. The Apollo/Orion spherical
segment is the limiting case theta_c = arccos(R_b / R_n), where the conical
frustum has zero length.

Fixed-topology grid: the panel count never changes with the design variables,
so the coefficients move smoothly as the shape moves.
"""
import numpy as np

GAMMA = 1.4


def cp_max(mach=np.inf, gamma=GAMMA):
    """Stagnation pressure coefficient behind a normal shock (Rayleigh pitot).

    The M -> inf limit is 1.8394 for gamma = 1.4. That limit is Oswatitsch
    Mach-number independence: above roughly M = 8 the coefficients stop
    changing with Mach.
    """
    if not np.isfinite(mach):
        # limit form as M -> inf
        return (2.0 / gamma) * (
            ((gamma + 1.0) ** 2 / (4.0 * gamma)) ** (gamma / (gamma - 1.0))
            * (2.0 * gamma / (gamma + 1.0))
        )
    m2 = mach ** 2
    a = ((gamma + 1.0) ** 2 * m2 / (4.0 * gamma * m2 - 2.0 * (gamma - 1.0))) \
        ** (gamma / (gamma - 1.0))
    b = (1.0 - gamma + 2.0 * gamma * m2) / (gamma + 1.0)
    return 2.0 / (gamma * m2) * (a * b - 1.0)


def meridian(R_n, theta_c_deg, R_b, n_cap=60, n_cone=40):
    """(x, r) profile points along the meridian, nose to base.

    Panel counts are fixed. Changing the shape moves the points but never
    changes how many there are.
    """
    th = np.radians(theta_c_deg)
    lam_max = 0.5 * np.pi - th                      # cap ends where it goes tangent
    lam = np.linspace(0.0, lam_max, n_cap + 1)
    x_cap = R_n * (1.0 - np.cos(lam))
    r_cap = R_n * np.sin(lam)

    if n_cone <= 0:
        return x_cap, r_cap
    r1 = R_n * np.cos(th)                           # tangency radius
    x1 = R_n * (1.0 - np.sin(th))
    # frustum from the tangency point out to the base
    t = np.linspace(0.0, 1.0, n_cone + 1)[1:]       # drop the duplicated joint
    r_cone = r1 + t * (R_b - r1)
    x_cone = x1 + (r_cone - r1) / np.tan(th)

    return np.concatenate([x_cap, x_cone]), np.concatenate([r_cap, r_cone])


def panels(R_n, theta_c_deg, R_b, n_cap=60, n_cone=40, n_theta=72):
    """Revolve the meridian and return panel centroids, area-weighted normals,
    and areas. Normals point outward."""
    x, r = meridian(R_n, theta_c_deg, R_b, n_cap, n_cone)
    phi = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)

    X = x[:, None] * np.ones_like(phi)[None, :]
    Y = r[:, None] * np.cos(phi)[None, :]
    Z = r[:, None] * np.sin(phi)[None, :]

    # quad corners
    p00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], axis=-1)
    p10 = np.stack([X[1:, :-1],  Y[1:, :-1],  Z[1:, :-1]],  axis=-1)
    p11 = np.stack([X[1:, 1:],   Y[1:, 1:],   Z[1:, 1:]],   axis=-1)
    p01 = np.stack([X[:-1, 1:],  Y[:-1, 1:],  Z[:-1, 1:]],  axis=-1)

    centroid = 0.25 * (p00 + p10 + p11 + p01)
    # area vector from the diagonal cross product; magnitude is the quad area
    d1 = p11 - p00
    d2 = p01 - p10
    area_vec = 0.5 * np.cross(d2, d1)   # order chosen so normals point outward
    area = np.linalg.norm(area_vec, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normal = area_vec / area[..., None]
    normal = np.nan_to_num(normal)                  # degenerate panel at the apex

    return (centroid.reshape(-1, 3), normal.reshape(-1, 3), area.reshape(-1))


def coefficients(R_n, theta_c_deg, R_b, alpha_deg=0.0, mach=np.inf,
                 A_ref=None, n_cap=60, n_cone=40, n_theta=72):
    """Modified Newtonian force coefficients, referenced to frontal area."""
    cen, nrm, area = panels(R_n, theta_c_deg, R_b, n_cap, n_cone, n_theta)
    if A_ref is None:
        A_ref = np.pi * R_b ** 2

    a = np.radians(alpha_deg)
    u = np.array([np.cos(a), 0.0, np.sin(a)])       # freestream direction

    s = -(nrm @ u)                                  # >0 on the windward side
    Cp = cp_max(mach) * np.maximum(0.0, s) ** 2     # C1-continuous clamp

    # force = -integral Cp n dA  (pressure acts inward along -n)
    C_A = -np.sum(Cp * nrm[:, 0] * area) / A_ref
    C_N = -np.sum(Cp * nrm[:, 2] * area) / A_ref

    C_D = C_A * np.cos(a) + C_N * np.sin(a)
    C_L = C_N * np.cos(a) - C_A * np.sin(a)
    return dict(C_A=C_A, C_N=C_N, C_D=C_D, C_L=C_L,
                LoD=C_L / C_D if C_D else np.nan, A_ref=A_ref)


def closed_form_CA(R_n, theta_c_deg, R_b, mach=np.inf):
    """Analytic Newtonian axial-force coefficient for a sphere-cone at alpha=0,
    referenced to pi*R_b^2.

    Spherical cap, integrating Cp_max cos^4(lam) sin(lam) over the ring:
        C_A,cap  = Cp_max (R_n^2 / 2 R_b^2) (1 - sin^4 theta_c)
    Conical frustum, where Cp is the constant Cp_max sin^2(theta_c):
        C_A,cone = Cp_max sin^2(theta_c) (1 - (R_n/R_b)^2 cos^2(theta_c))

    This is the Grant & Braun style closed form and is exact, so it is the
    ground truth the panel model has to reproduce.
    """
    th = np.radians(theta_c_deg)
    cap = (R_n ** 2 / (2.0 * R_b ** 2)) * (1.0 - np.sin(th) ** 4)
    cone = np.sin(th) ** 2 * (1.0 - (R_n / R_b) ** 2 * np.cos(th) ** 2)
    return cp_max(mach) * (cap + cone)


def theta_c_for_spherical_segment(R_n, R_b):
    """Cone half-angle at which the frustum vanishes and the body is a pure
    spherical segment — the Apollo/Orion heatshield case."""
    return np.degrees(np.arccos(R_b / R_n))


if __name__ == "__main__":
    print(f"Cp_max (M -> inf, gamma=1.4) = {cp_max():.6f}")
    print(f"Cp_max (M = 20)              = {cp_max(20.0):.6f}")
    print(f"Cp_max (M = 8)               = {cp_max(8.0):.6f}")
    print()

    print("SANITY: hemisphere should give exactly Cp_max/2")
    hemi = coefficients(1.0, 0.0 + 1e-9, 1.0, n_cap=400, n_cone=1, n_theta=240)
    print(f"  panel C_D = {hemi['C_D']:.6f}   exact = {cp_max()/2:.6f}")
