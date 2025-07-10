class EinsteinFinalEntropic(EinsteinFinalBase):
    r"""Models gravity as an entropic force, modifying the potential with a logarithmic term.
    <reason>Inspired by theories of emergent gravity (e.g., Verlinde), this model modifies the gravitational potential based on thermodynamic and holographic principles, where gravity arises from information entropy.</reason>"""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Entropic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * LP**2 / r**2 * torch.log(r / LP)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
