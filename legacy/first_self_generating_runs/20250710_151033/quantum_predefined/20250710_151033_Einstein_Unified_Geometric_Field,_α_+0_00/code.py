class EinsteinUnifiedGeometricField(EinsteinFinalBase):
    r"""A candidate for Einstein's final theory, synthesizing his work on unification.
    <reason>This theory represents a culmination of the project's goals. It is a creative, physically motivated attempt to model the principles of unification that Einstein pursued. It combines his known theoretical approaches into a single, testable hypothesis.</reason>"""
    sweep = None

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Unified Geometric Field, α={alpha:+.2f}")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        u_g = rs / r
        u_e = rq_sq / r**2
        log_mod = self.alpha * torch.log1p(u_g)
        unified_potential = u_g - (u_e / (1 + u_g))
        g_tt = -(1 - unified_potential + log_mod)
        g_rr = 1 / (1 - unified_potential - log_mod + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)
