class EinsteinFinalUnifiedAdditive(EinsteinFinalBase):
    """A test of unified theory where the EM field is added with a variable coupling."""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Unified Additive, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        m = 1 - rs/r + self.alpha * (rq_sq / r**2)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
