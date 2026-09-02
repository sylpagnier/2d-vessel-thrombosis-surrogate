import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from src.config import PhysicsConfig

# Grid on which the 1D profile is tabulated before it is splined.  The profile is a smooth
# function of y alone, so a cubic spline over this many points carries a relative error of
# order (1/N)^4 ~ 1e-13 -- four orders below the 1e-9 the FEM Picard loop is asked to reach.
_PROFILE_GRID = 2001
# Bisection sweeps for the vectorised shear-rate inversion.  The bracket is
# [tau/mu_0, tau/mu_inf], a factor of mu_0/mu_inf ~ 10 wide in log space, so 60 halvings
# already reach double precision; 80 is free because the sweep is one array op per halving.
_BISECT_ITERS = 80


def shear_stress(gamma, mu_inf, mu_0, lam, n, a):
    """Calculate shear stress for Carreau fluid."""
    mu = mu_inf + (mu_0 - mu_inf) * (1 + (lam * gamma)**a)**((n - 1) / a)
    return mu * gamma


def find_gamma(tau, mu_inf, mu_0, lam, n, a):
    """Find shear rate gamma given shear stress tau."""
    if tau == 0:
        return 0.0
    # gamma must be positive, and tau is monotonically increasing with gamma
    # upper bound: tau / mu_inf (since mu >= mu_inf)
    gamma_max = tau / mu_inf
    gamma_min = tau / mu_0

    def obj(gamma):
        return shear_stress(gamma, mu_inf, mu_0, lam, n, a) - tau

    res = root_scalar(obj, bracket=[gamma_min, gamma_max], method='brentq')
    return res.root


def find_gamma_vec(tau, mu_inf, mu_0, lam, n, a):
    """`find_gamma` over a whole array, by bisection in log(gamma).

    The scalar version calls `root_scalar`/brentq once per point.  The inlet condition needs
    the shear rate at tens of thousands of points, and driving a Python-level Brent solve
    through each of them was 54% of the entire FEM solve wall clock -- for a function of one
    variable that is the same at every point.

    tau(gamma) = mu(gamma) * gamma is strictly increasing, and mu is bounded between mu_inf
    and mu_0, so [tau/mu_0, tau/mu_inf] brackets the root for every tau > 0.  Bisecting that
    bracket in log space converges monotonically for the whole array at once, with no
    per-point control flow, and lands on the same root brentq does to double precision.
    """
    tau = np.asarray(tau, dtype=np.float64)
    out = np.zeros_like(tau)
    pos = tau > 0.0
    if not np.any(pos):
        return out
    t = tau[pos]
    lo = np.log(t / mu_0)
    hi = np.log(t / mu_inf)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        g = np.exp(mid)
        f = shear_stress(g, mu_inf, mu_0, lam, n, a) - t
        neg = f < 0.0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    out[pos] = np.exp(0.5 * (lo + hi))
    return out


def _carreau_profile_spline(U_inlet, H, mu_inf, mu_0, lam, n_car, a):
    """Tabulate the fully developed Carreau profile u(y) on [0, H/2] and return its spline.

    The physics is unchanged from the quadrature version this replaces:

        tau(y) = G * y,   gamma(y) = tau^-1(tau(y)),   u(y) = int_y^{H/2} gamma(y') dy'

    with the pressure gradient G fixed by requiring the mean velocity to equal `U_inlet`.
    What changes is that gamma is now evaluated on one shared grid and integrated by the
    analytic antiderivative of its cubic spline, instead of handing `scipy.integrate.quad`
    an adaptive integral -- with a nested root find inside the integrand -- separately for
    every single query point.
    """
    half = 0.5 * H
    ys = np.linspace(0.0, half, _PROFILE_GRID)

    def _gamma_grid(G):
        return find_gamma_vec(G * ys, mu_inf, mu_0, lam, n_car, a)

    def _mean_velocity(G):
        # (2/H) * int_0^{H/2} gamma(y) y dy, the same identity the quad version integrated:
        # integrating u(y) over the half channel by parts turns it into this moment of gamma.
        return (2.0 / H) * CubicSpline(ys, _gamma_grid(G) * ys).antiderivative()(half)

    # Newtonian limits bracket G: mu_inf <= mu(gamma) <= mu_0 makes the Carreau channel need
    # a pressure gradient between the two Poiseuille values for the same mean velocity.
    G_min = 12 * mu_inf * U_inlet / (H**2)
    G_max = 12 * mu_0 * U_inlet / (H**2)
    G_sol = root_scalar(lambda G: _mean_velocity(G) - U_inlet,
                        bracket=[G_min, G_max], method='brentq').root

    # u(y) = int_y^{H/2} gamma = Gam(H/2) - Gam(y) for any antiderivative Gam of gamma.
    gam = CubicSpline(ys, _gamma_grid(G_sol)).antiderivative()
    top = float(gam(half))
    return lambda y_abs: top - gam(y_abs)


def get_inlet_profile(U_inlet, H, y_coords, config: PhysicsConfig = None, recenter=False):
    """
    Generate fully developed 1D Carreau flow profile.
    
    Args:
        U_inlet: Average velocity across the channel.
        H: Channel width.
        y_coords: Array of y-coordinates along the inlet (arbitrary orientation, but relative to a wall).
                  Assuming the channel spans from y_min to y_max. We will recenter to [-H/2, H/2].
        config: PhysicsConfig containing rheology parameters.
    """
    if config is None:
        config = PhysicsConfig()
        
    mu_inf = config.mu_inf
    mu_0 = config.mu_0
    lam = config.lam
    n_car = config.n
    a = config.a
    
    if config.viscosity_model == "newtonian":
        # Parabolic profile
        if recenter:
            y_center = np.mean([np.min(y_coords), np.max(y_coords)])
            y_norm = (y_coords - y_center) / (H / 2)
        else:
            y_norm = y_coords / (H / 2)
        return 1.5 * U_inlet * (1 - y_norm**2)

    y_coords = np.asarray(y_coords, dtype=np.float64)
    if recenter:
        # Recenter y_coords
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        y_center = (y_min + y_max) / 2
        y_shifted = y_coords - y_center
    else:
        y_shifted = y_coords

    if U_inlet == 0.0:
        return np.zeros_like(y_shifted)

    profile = _carreau_profile_spline(U_inlet, H, mu_inf, mu_0, lam, n_car, a)

    half = 0.5 * H
    y_abs = np.abs(y_shifted)
    # Outside the channel the profile is identically zero; clip before evaluating so the
    # spline is never extrapolated past the grid it was built on.
    outside = y_abs >= half
    u_vals = profile(np.where(outside, half, y_abs))
    return np.where(outside, 0.0, u_vals)
