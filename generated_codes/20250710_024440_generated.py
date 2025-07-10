```python
class EinsteinFinalUnifiedAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinFinalUnifiedAlpha0_5")
        self.alpha = 0.5

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / (C_param ** 2)
        A = 1 - rs / r + self.alpha * (rs / r) ** 2
        g_tt = - (C_param ** 2) * A
        g_rr = 1.0 / A
        g_thetatheta = r ** 2
        g_phiphi = r ** 2
        return g_tt, g_rr, g_thetatheta, g_phiphi
```