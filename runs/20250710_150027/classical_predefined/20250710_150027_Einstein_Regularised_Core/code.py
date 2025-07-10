class EinsteinRegularized(GravitationalTheory):
    """
    A regularized version of GR that avoids a central singularity.
    <reason>This model is a key 'distinguishable' theory. It modifies GR only at the Planck scale, and its tiny but non-zero loss demonstrates the framework's extreme sensitivity to subtle physical deviations.</reason>
    """
    category = "classical"
    sweep = None

    def __init__(self):
        super().__init__("Einstein Regularised Core")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / torch.sqrt(r**2 + LP**2)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
