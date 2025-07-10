class QuantumCorrected(GravitationalTheory):
    """
    A generic model with a cubic correction term, representing some quantum effects.
    <reason>This model serves as a simple test case for higher-order corrections to the GR metric. Its performance relative to other theories helps classify the nature of potential quantum gravitational effects.</reason>
    """
    category = "quantum"
    sweep = dict(alpha=np.linspace(-2.0, 2.0, 9))

    def __init__(self, alpha: float):
        super().__init__(f"Quantum Corrected (α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + self.alpha * (rs / r) ** 3
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
