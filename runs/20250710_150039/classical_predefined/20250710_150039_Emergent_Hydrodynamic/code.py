class EmergentHydrodynamic(GravitationalTheory):
    """
    An emergent model treating gravity as hydrodynamic flow.
    <reason>Re-introduced from paper (Section 4.2) for its high loss in dynamics, validating the framework's sensitivity to incorrect geometries.</reason>
    """
    category = "classical"
    sweep = None

    def __init__(self):
        super().__init__("Emergent (Hydrodynamic)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Hydrodynamic approximation: velocity-like term
        flow_term = 0.05 * torch.sqrt(rs / r)
        m = 1 - rs / r - flow_term
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
