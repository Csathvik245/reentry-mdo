"""
Modified Newtonian aerodynamics in CSDL — differentiable with respect to the
geometry, so derivatives flow straight through to the optimizer.

    Cp = Cp_max * relu(-n.u)^2

relu(x)^2 is C1-continuous: its derivative 2*relu(x) is continuous everywhere,
including at the shadow boundary x = 0. So the windward/leeward transition is
smooth and needs no special handling on a convex forebody.

The grid has fixed topology. Panel counts never change with the design
variables; only the vertex positions move. That is what keeps the coefficients
smooth as the shape changes.
"""
import numpy as np
import csdl_alpha as csdl

GAMMA = 1.4
CPMAX_INF = (2.0 / GAMMA) * (
    ((GAMMA + 1.0) ** 2 / (4.0 * GAMMA)) ** (GAMMA / (GAMMA - 1.0))
    * (2.0 * GAMMA / (GAMMA + 1.0))
)   # 1.83937, the Oswatitsch Mach-independent limit


def sphere_cone_surface(R_n, theta_c, R_b, n_cap=40, n_cone=20, n_theta=80):
    """Revolved sphere-cone surface as CSDL variables.

    R_n, R_b     : CSDL scalars (nose radius, base radius)
    theta_c      : CSDL scalar, cone half-angle in RADIANS
    Returns X, Y, Z each of shape (n_cap + n_cone + 1, n_theta + 1).
    """
    lam_max = np.pi / 2.0 - theta_c                 # cap ends at tangency
    frac = np.linspace(0.0, 1.0, n_cap + 1)
    lam = csdl.outer(csdl.Variable(value=frac), lam_max)
    lam = csdl.reshape(lam, (n_cap + 1,))

    x_cap = R_n * (1.0 - csdl.cos(lam))
    r_cap = R_n * csdl.sin(lam)

    if n_cone > 0:
        r1 = R_n * csdl.cos(theta_c)
        x1 = R_n * (1.0 - csdl.sin(theta_c))
        t = np.linspace(0.0, 1.0, n_cone + 1)[1:]   # drop the duplicated joint
        tv = csdl.Variable(value=t)
        r_cone = csdl.expand(r1, (n_cone,)) \
            + tv * csdl.expand(R_b - r1, (n_cone,))
        x_cone = csdl.expand(x1, (n_cone,)) \
            + (r_cone - csdl.expand(r1, (n_cone,))) / csdl.expand(
                csdl.tan(theta_c), (n_cone,))
        x = csdl.concatenate([x_cap, x_cone])
        r = csdl.concatenate([r_cap, r_cone])
    else:
        x, r = x_cap, r_cap

    phi = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    ones_phi = csdl.Variable(value=np.ones(n_theta + 1))
    X = csdl.outer(x, ones_phi)
    Y = csdl.outer(r, csdl.Variable(value=np.cos(phi)))
    Z = csdl.outer(r, csdl.Variable(value=np.sin(phi)))
    return X, Y, Z


def newtonian_coefficients(R_n, theta_c, R_b, alpha, A_ref=None,
                           cp_max=CPMAX_INF,
                           n_cap=40, n_cone=20, n_theta=80):
    """Modified Newtonian C_A, C_N, C_D, C_L.

    alpha is a CSDL scalar in RADIANS. A_ref defaults to pi * R_b^2.
    """
    X, Y, Z = sphere_cone_surface(R_n, theta_c, R_b, n_cap, n_cone, n_theta)
    if A_ref is None:
        A_ref = np.pi * R_b ** 2

    def corners(A):
        return (A[:-1, :-1], A[1:, :-1], A[1:, 1:], A[:-1, 1:])

    X00, X10, X11, X01 = corners(X)
    Y00, Y10, Y11, Y01 = corners(Y)
    Z00, Z10, Z11, Z01 = corners(Z)

    # the two quad diagonals
    d1x, d1y, d1z = X11 - X00, Y11 - Y00, Z11 - Z00
    d2x, d2y, d2z = X01 - X10, Y01 - Y10, Z01 - Z10

    # area vector = 0.5 * d2 x d1  (this order gives outward normals)
    ax = 0.5 * (d2y * d1z - d2z * d1y)
    ay = 0.5 * (d2z * d1x - d2x * d1z)
    az = 0.5 * (d2x * d1y - d2y * d1x)

    ux, uz = csdl.cos(alpha), csdl.sin(alpha)       # freestream direction
    sh = ax.shape
    dot = -(ax * csdl.expand(ux, sh) + az * csdl.expand(uz, sh))   # = -a.u
    aa = ax ** 2 + ay ** 2 + az ** 2                # |a|^2, no square root

    # What we actually need is Cp * n * dA, and n dA is the area vector a:
    #     Cp * a = cp_max * max(0, -a.u / |a|)^2 * a
    #            = cp_max * max(0, -a.u)^2 / |a|^2 * a
    # Writing it this way removes the square root, whose derivative blows up
    # wherever a panel has near-zero area. That matters because the conical
    # frustum has exactly zero length in the spherical-segment limit
    # (theta_c = arccos(R_b/R_n)), which makes every cone panel degenerate.
    #
    # max(0,x) = (x + |x|)/2 with |x| = sqrt(x^2 + eps). csdl.relu cannot be
    # used here: it returns a NaN derivative at exactly x = 0, and panels do
    # sit exactly on the shadow boundary.
    scale = csdl.expand(R_b ** 2, sh)               # natural area scale
    eps_dot = 1e-30 * scale ** 2
    eps_aa = 1e-16 * scale ** 2
    m = 0.5 * (dot + (dot ** 2 + eps_dot) ** 0.5)   # smoothed max(0, dot)
    Cp_over_aa = cp_max * m ** 2 / (aa + eps_aa)

    # force = -int Cp n dA
    C_A = -csdl.sum(Cp_over_aa * ax) / A_ref
    C_N = -csdl.sum(Cp_over_aa * az) / A_ref

    ca, sa = csdl.cos(alpha), csdl.sin(alpha)
    C_D = C_A * ca + C_N * sa
    C_L = C_N * ca - C_A * sa
    return C_A, C_N, C_D, C_L


def theta_c_tangent(R_n, R_b):
    """Cone half-angle at which the frustum vanishes — the pure spherical
    segment (Apollo/Orion heatshield) case. Returns radians."""
    return csdl.arccos(R_b / R_n)
