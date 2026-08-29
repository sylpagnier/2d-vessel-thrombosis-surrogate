import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
from src.config import PhysicsConfig

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
        
    # Carreau Profile
    def gamma_of_y(y, G):
        tau = G * y
        return find_gamma(tau, mu_inf, mu_0, lam, n_car, a)
        
    def average_velocity(G):
        # Integral of gamma(y) * y from 0 to H/2
        integral, _ = quad(lambda y: gamma_of_y(y, G) * y, 0, H/2)
        return (2 / H) * integral
        
    # Find pressure gradient G
    def obj_G(G):
        return average_velocity(G) - U_inlet
        
    # Bounds for G
    # Newtonian limits: G_newt = 12 * mu * U_inlet / H^2
    G_min = 12 * mu_inf * U_inlet / (H**2)
    G_max = 12 * mu_0 * U_inlet / (H**2)
    
    res = root_scalar(obj_G, bracket=[G_min, G_max], method='brentq')
    G_sol = res.root
    
    # Precompute velocity profile by integrating gamma from wall to y
    # u(y) = int_{y}^{H/2} gamma(y') dy'
    def velocity(y):
        y_abs = np.abs(y)
        if y_abs >= H/2:
            return 0.0
        val, _ = quad(lambda y_prime: gamma_of_y(y_prime, G_sol), y_abs, H/2)
        return val
        
    if recenter:
        # Recenter y_coords
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        y_center = (y_min + y_max) / 2
        y_shifted = y_coords - y_center
    else:
        y_shifted = y_coords
    
    y_flat = y_shifted.flatten()
    u_vals = np.array([velocity(y) for y in y_flat]).reshape(y_shifted.shape)
    return u_vals
