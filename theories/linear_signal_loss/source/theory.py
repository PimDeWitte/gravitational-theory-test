from base_theory import GravitationalTheory, Tensor
import torch
import numpy as np

class LinearSignalLoss(GravitationalTheory):
    """
    Introduces a parameter that smoothly degrades the gravitational signal as a function of proximity to the central mass.
    <reason>Re-introduced from paper (Section 3.1) as a promising model to measure breaking points in informational fidelity, analogous to compression quality degradation.</reason>
    """
    category = "classical"
    sweep = dict(gamma=np.linspace(0.0, 1.0, 5))
    cacheable = True

    def __init__(self, gamma: float = 0.0):
        super().__init__(f"Linear Signal Loss (γ={gamma:+.2f})")
        # Don't force device - let the framework handle it
        self.gamma = gamma

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Ensure gamma is on same device as r
        gamma = torch.tensor(self.gamma, device=r.device, dtype=r.dtype)
        
        # <reason>Fix metric to match feedback math: m = (1 - γ rs/r)(1 - rs/r) = 1 - (1+γ)rs/r + γ(rs/r)^2</reason>
        # This expansion naturally produces RN-like behavior at γ=0.75
        rs_over_r = rs / r
        m = (1 - gamma * rs_over_r) * (1 - rs_over_r)
        
        # Equivalent to: m = 1 - (1 + gamma) * rs_over_r + gamma * rs_over_r**2
        # At γ=0.75, this mimics RN's quadratic term for unification
        
        # Add small epsilon only for numerical stability in denominators
        epsilon = 1e-10
        g_tt = -m
        g_rr = 1 / (m + epsilon)
        g_pp = r**2
        g_tp = torch.zeros_like(r)
        
        return g_tt, g_rr, g_pp, g_tp 