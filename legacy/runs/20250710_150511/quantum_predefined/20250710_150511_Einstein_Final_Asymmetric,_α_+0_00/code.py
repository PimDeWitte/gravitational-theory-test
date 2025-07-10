class EinsteinFinalAsymmetric(EinsteinFinalBase):
    """Simulates an asymmetric metric by modifying g_tt and g_rr differently."""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Asymmetric, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        mod = self.alpha * (rs/r)**2
        g_tt = -(1 - rs/r + mod)
        g_rr = 1 / (1 - rs/r - mod + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)
