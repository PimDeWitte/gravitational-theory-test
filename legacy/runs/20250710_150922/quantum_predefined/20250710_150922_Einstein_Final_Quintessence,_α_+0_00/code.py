class EinsteinFinalQuintessence(EinsteinFinalBase):
    r"""A model that includes a quintessence-like scalar field.
    <reason>Quintessence is a hypothesized form of dark energy. This models its effect on local spacetime geometry as a very shallow power-law term, distinct from a cosmological constant.</reason>"""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Quintessence, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r - self.alpha * (r/rs)**0.5
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
