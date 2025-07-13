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
        degradation = gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + 1e-10))
        return -m, 1 / (m + 1e-10), r**2, torch.zeros_like(r) 