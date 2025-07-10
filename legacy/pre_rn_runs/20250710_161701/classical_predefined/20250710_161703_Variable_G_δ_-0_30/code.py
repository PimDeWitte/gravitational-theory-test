class VariableG(GravitationalTheory):
    """
    A model where the gravitational constant G varies with distance.
    <reason>This theory tests the fundamental assumption of a constant G. The asymmetric failure (stable for weakening G, unstable for strengthening G) provides a powerful insight into the necessary conditions for a stable universe.</reason>
    """
    category = "classical"
    sweep = dict(delta=np.linspace(-0.5, 0.1, 7))
    cacheable = True

    def __init__(self, delta: float):
        super().__init__(f"Variable G (δ={delta:+.2f})")
        self.delta = torch.as_tensor(delta, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        G_eff = G_param * (1 + self.delta * torch.log1p(r / rs))
        m = 1 - 2 * G_eff * M_param / (C_param**2 * r)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
