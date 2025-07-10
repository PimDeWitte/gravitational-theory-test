```python
import torch
from torch import Tensor

class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / (C_param ** 2)
        alpha = 0.5
        A = 1 - rs / r + alpha * torch.pow(rs / r, 2)
        g_tt = - (C_param ** 2) * A
        g_rr = 1 / A
        g_theta_theta = r ** 2
        g_phi_phi = r ** 2
        return g_tt, g_rr, g_theta_theta, g_phi_phi
```