from base_theory import GravitationalTheory, Tensor
import torch
import math
from scipy.constants import epsilon_0, G, c

class ReissnerNordstrom(GravitationalTheory):
    """
    The Reissner-Nordström metric - exact solution for a charged, non-rotating mass.
    This is our baseline for gravity + electromagnetism.
    """
    category = "classical"
    cacheable = True

    def __init__(self, Q: float):
        super().__init__(f"Reissner-Nordström (Q={Q:.2e})")
        self.Q = Q
        # Precompute rq_sq in Python float64 to avoid overflow when Q is large
        # This prevents numerical issues when Q=1e20 (Q^2 = 1e40)
        self.rq_sq_precomputed = (G * Q**2) / (4 * math.pi * epsilon_0 * c**4)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Use precomputed value and convert to tensor with same device/dtype as r
        rq_sq = torch.as_tensor(self.rq_sq_precomputed, device=r.device, dtype=r.dtype)
        m = 1 - rs / r + rq_sq / r**2
        # Use a more robust epsilon that scales with the dtype
        epsilon = torch.finfo(r.dtype).eps * 100
        return -m, 1 / (m + epsilon), r**2, torch.zeros_like(r)
    
    def get_cache_tag(self, N_STEPS, precision_tag, r0_tag):
        """Include Q parameter in cache tag."""
        base = self.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(".", "_")
        Q_tag = f"Q{self.Q:.0e}".replace("+", "p").replace("-", "m")
        return f"{base}_{Q_tag}_{N_STEPS}_{precision_tag}_r{r0_tag}" 