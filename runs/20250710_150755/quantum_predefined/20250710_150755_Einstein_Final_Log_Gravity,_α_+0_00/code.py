class EinsteinFinalLogGravity(EinsteinFinalBase):
    """A logarithmic modification to the gravitational potential."""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Log Gravity, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r) * (1 - self.alpha * torch.log1p(rs/r))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
