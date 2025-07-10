class EinsteinFinalHigherDeriv(EinsteinFinalBase):
    r"""A model with both quadratic and cubic corrections.
    <reason>Instead of testing just one higher-order term, this model includes two, allowing for more complex interactions and a better fit if the true quantum corrections are not simple power laws.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Higher-Derivative, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**2 - self.alpha * (rs/r)**3
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
