class ReissnerNordstrom(GravitationalTheory):
    """
    The Reissner-Nordström metric for a charged, non-rotating black hole.
    <reason>This is the exact solution for a charged mass and serves as the second ground truth (the Kaluza-Klein baseline) for testing a theory's ability to unify gravity and electromagnetism.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self, Q: float):
        super().__init__(f"Reissner‑Nordström (Q={Q:.1e})")
        self.Q = Q  # Keep as Python float to avoid dtype casting issues
        # <reason>chain: Changed to store Q as float to enable safe computation of large values in f64 before casting to DTYPE.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        q_scaled = self.Q / C_param**2
        # <reason>chain: Scaled Q by C_param**2 first to reduce magnitude and prevent overflow when squaring in subsequent steps.</reason>
        rq_sq = (G_param * (q_scaled ** 2)) / (4 * TORCH_PI * EPS0_T * r**2)
        # <reason>chain: Computed rq_sq using the scaled Q to ensure all intermediate values stay within float32 limits.</reason>
        m = 1 - rs / r + rq_sq / r**2
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
