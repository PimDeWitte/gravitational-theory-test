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
        self.Q = Q
        # Precompute rq_sq in Python float (f64) to avoid overflow in f32
        rq_sq_py = (G * (Q ** 2)) / (4 * math.pi * epsilon_0 * (c ** 4))
        self.rq_sq = torch.as_tensor(rq_sq_py, device=device, dtype=DTYPE)
        # <reason>chain: Precomputed rq_sq using Python doubles to handle large Q**2 without overflow, then convert to tensor with DTYPE.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # No q_scaled needed now
        m = 1 - rs / r + self.rq_sq / r**2
        # <reason>chain: Used precomputed rq_sq directly in m, removing incorrect / r**2 from rq_sq and eliminating scaling to restore standard RN metric.</reason>
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
