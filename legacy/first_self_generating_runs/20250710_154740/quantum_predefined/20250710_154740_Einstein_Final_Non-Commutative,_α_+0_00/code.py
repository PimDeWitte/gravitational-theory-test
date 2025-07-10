class EinsteinFinalNonCommutative(EinsteinFinalBase):
    r"""A model motivated by non-commutative geometry, which regularizes the singularity.
    <reason>Non-commutative geometry suggests that spacetime coordinates do not commute at the Planck scale, which effectively 'smears' the singularity. This is modeled by an exponential term that smooths the metric core.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Non-Commutative, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs * torch.exp(-self.alpha * LP**2 / r**2)) / r
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
