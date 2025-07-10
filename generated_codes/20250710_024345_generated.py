```python
class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / (C_param ** 2)
        alpha = 0.5
        one = torch.ones_like(r)
        A = one - rs / r + alpha * (rs / r) ** 2
        g_tt = -A * (C_param ** 2)
        g_rr = one / A
        g_thth = r ** 2
        g_phiphi = r ** 2
        return g_tt, g_rr, g_thth, g_phiphi
```