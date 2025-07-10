class LogCorrected(GravitationalTheory):
    """
    A quantum gravity inspired model with a logarithmic correction term.
    <reason>This model is a high-performing 'distinguishable'. Logarithmic corrections are predicted by some quantum loop gravity theories, making this a promising candidate for a first-order quantum correction to GR.</reason>
    """
    category = "quantum"
    sweep = dict(beta=np.linspace(-0.50, 0.17, 7))

    def __init__(self, beta: float):
        super().__init__(f"Log Corrected (β={beta:+.2f})")
        self.beta = torch.as_tensor(beta, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        sr = torch.maximum(r, rs * 1.001)
        log_corr = self.beta * (rs / sr) * torch.log(sr / rs)
        m = 1 - rs / r + log_corr
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
