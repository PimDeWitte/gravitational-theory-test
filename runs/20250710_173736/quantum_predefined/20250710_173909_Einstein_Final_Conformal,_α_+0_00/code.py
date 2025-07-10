class EinsteinFinalConformal(EinsteinFinalBase):
    r"""A model inspired by conformal gravity, where physics is invariant under scale transformations.
    <reason>Conformal gravity is an alternative to GR that has different properties at cosmological scales. This model introduces a term that respects conformal symmetry, testing a different geometric foundation.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Conformal, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * r
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
