class EinsteinFinalMembrane(EinsteinFinalBase):
    r"""A correction inspired by higher-dimensional brane-world scenarios.
    <reason>In some string theory models, our universe is a 'brane' in a higher-dimensional space. This can lead to gravity 'leaking' into other dimensions, which is modeled here as a steep correction to the potential.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Membrane, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - torch.sqrt((rs/r)**2 + self.alpha * (LP/r)**4)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
