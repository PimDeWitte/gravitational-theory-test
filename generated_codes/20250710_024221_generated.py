```python
from torch import Tensor
import torch

class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = 0.5
        rs = 2 * G_param * M_param / C_param**2
        correction = alpha * torch.pow(rs / r, 2) * (1 - rs / r)
        term = 1 - rs / r + correction
        g_tt = -term
        g_rr = 1 / term
        g_theta_theta = r**2
        g_phi_phi = r**2
        return g_tt, g_rr, g_theta_theta, g_phi_phi
```