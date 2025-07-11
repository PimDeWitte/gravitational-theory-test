
# --- CLASSICAL THEORIES ---

class Schwarzschild(GravitationalTheory):
    """
    The Schwarzschild metric for a non-rotating, uncharged black hole.
    <reason>This is the exact solution to Einstein's field equations in a vacuum and serves as the fundamental ground truth (baseline) for pure gravity in this framework.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self):
        super().__init__("Schwarzschild (GR)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / (r + EPSILON)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class NewtonianLimit(GravitationalTheory):
    """
    The Newtonian approximation of gravity.
    <reason>This theory is included as a 'distinguishable' model. It correctly lacks spatial curvature (g_rr = 1), and its significant but finite loss value validates the framework's ability to quantify physical incompleteness.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self):
        super().__init__("Newtonian Limit")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        return -m, torch.ones_like(r), r**2, torch.zeros_like(r)

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

class LinearSignalLoss(GravitationalTheory):
    """
    Introduces a parameter that smoothly degrades the gravitational signal as a function of proximity to the central mass.
    <reason>Re-introduced from paper (Section 3.1) as a promising model to measure breaking points in informational fidelity, analogous to compression quality degradation.</reason>
    """
    category = "classical"
    sweep = dict(gamma=np.linspace(0.0, 1.0, 5))
    cacheable = True

    def __init__(self, gamma: float):
        super().__init__(f"Linear Signal Loss (γ={gamma:+.2f})")
        self.gamma = torch.as_tensor(gamma, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + EPSILON))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
