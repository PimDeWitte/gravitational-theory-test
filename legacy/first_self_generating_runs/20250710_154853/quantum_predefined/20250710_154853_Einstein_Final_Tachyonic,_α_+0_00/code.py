class EinsteinFinalTachyonic(EinsteinFinalBase):
    r"""A speculative model with a tachyonic field contribution.
    <reason>Tachyonic fields, while problematic, appear in some string theory contexts. This model tests the effect of a potential that weakens at short distances, a hallmark of tachyon condensation.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Tachyonic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r * (1 - self.alpha * torch.tanh(rs/r))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
