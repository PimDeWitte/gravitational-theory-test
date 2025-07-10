import numpy as np

class GravitationalTheory:
    """
    Base class for a physical theory. Each theory must be able to return
    the components of its spacetime metric at a given radius.
    """
    def __init__(self, name):
        """Initializes the model with a human-readable name."""
        self.name = name

    def get_metric(self, r, M, C, G, **kwargs):
        """
        Calculates g_tt and g_rr. Must be implemented by each specific theory.
        """
        raise NotImplementedError("Each theory must implement its own get_metric method.")

    def get_metric_derivatives(self, r, M, C, G, h=1e-4, **kwargs):
        """
        Numerically calculates the derivatives of the metric components, which are
        required for the geodesic equations.
        """
        g_tt_plus, g_rr_plus = self.get_metric(r + h, M, C, G, **kwargs)
        g_tt_minus, g_rr_minus = self.get_metric(r - h, M, C, G, **kwargs)
        
        g_tt_prime = (g_tt_plus - g_tt_minus) / (2 * h)
        g_rr_prime = (g_rr_plus - g_rr_minus) / (2 * h)
        
        return g_tt_prime, g_rr_prime

def geodesic_ode(tau, y, model, M, C, G):
    """
    Defines the system of first-order Ordinary Differential Equations for a geodesic.
    This is the core physics engine, passed to a numerical solver.
    The state vector y = [t, r, phi, dt/dtau, dr/dtau, dphi/dtau].
    """
    t, r, phi, dt, dr, dphi = y

    g_tt, g_rr = model.get_metric(r, M, C, G)
    g_tt_prime, g_rr_prime = model.get_metric_derivatives(r, M, C, G)

    if np.isinf(g_rr) or r <= (2 * G * M) / (C**2):
        return [0, 0, 0, 0, 0, 0]

    d2t = - (g_tt_prime / g_tt) * dr * dt
    d2r = - (0.5 / g_rr) * (g_tt_prime * dt**2 + g_rr_prime * dr**2 - 2 * r * dphi**2)
    d2phi = - (2.0 / r) * dr * dphi
    
    return [dt, dr, dphi, d2t, d2r, d2phi]