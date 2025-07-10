class EinsteinFinalGaussBonnet(EinsteinFinalBase):
    r"""A simplified model inspired by Gauss-Bonnet gravity, a common extension to GR.
    <reason>Gauss-Bonnet gravity adds a specific quadratic curvature term to the action. This phenomenological model captures the essence of such a modification with a steep 1/r^5 term that can arise in the metric solution.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Gauss-Bonnet, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**5
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
