class EinsteinFinalPowerLaw(EinsteinFinalBase):
    r"""Generalizes the potential with a variable power law, deviating from 1/r.
    <reason>This is a fundamental test of the inverse-square law at relativistic scales. By allowing the exponent to deviate from 1 (via alpha), we can test for large-scale modifications to gravity.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Power Law, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r)**(1.0 - self.alpha)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
