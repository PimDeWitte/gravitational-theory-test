class EinsteinFinalDilaton(EinsteinFinalBase):
    r"""A model including a dilaton field from string theory.
    <reason>String theory predicts the existence of a scalar field, the dilaton, which couples to gravity. This model tests a simple form of this coupling, modifying the strength of the gravitational potential.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Dilaton, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r) / (1 + self.alpha * (rs/r))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
