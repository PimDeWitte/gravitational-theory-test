class EinsteinFinalVacuum(EinsteinFinalBase):
    r"""A model where gravity's strength is coupled to the vacuum energy.
    <reason>This tests the idea that the strength of gravity could be affected by the energy density of the quantum vacuum, modeled here as a constant offset to the metric potential controlled by alpha.</reason>"""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Vacuum Coupling, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (LP/rs)**2
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
