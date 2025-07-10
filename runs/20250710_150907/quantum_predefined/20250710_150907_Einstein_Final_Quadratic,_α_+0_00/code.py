class EinsteinFinalQuadratic(EinsteinFinalBase):
    """A quadratic correction term, testing a different power-law deviation."""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Quadratic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**2
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
