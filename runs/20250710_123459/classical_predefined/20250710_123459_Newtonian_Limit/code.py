class NewtonianLimit(GravitationalTheory):
    """
    The Newtonian approximation of gravity.
    <reason>This theory is included as a 'distinguishable' model. It correctly lacks spatial curvature (g_rr = 1), and its significant but finite loss value validates the framework's ability to quantify physical incompleteness.</reason>
    """
    def __init__(self): super().__init__("Newtonian Limit")
    # <reason>chain: Initialized NewtonianLimit.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        return -m, torch.ones_like(r), r**2, torch.zeros_like(r)
    # <reason>chain: Defined Newtonian metric.</reason>
