class Participatory(GravitationalTheory):
    """
    A model where the metric is a weighted average of GR and flat spacetime, simulating observer participation.
    <reason>Re-introduced from paper (Section 4.3.1) as it demonstrates geometric brittleness; small deviations cause rapid degradation, highlighting GR's precision.</reason>
    """
    category = "classical"
    sweep = None

    def __init__(self, weight: float = 0.92):
        super().__init__(f"Participatory (w={weight:.2f})")
        self.weight = torch.as_tensor(weight, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m_gr = 1 - rs / (r + EPSILON)
        m_flat = torch.ones_like(r)
        m = self.weight * m_gr + (1 - self.weight) * m_flat
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
