class EinsteinDeathbedUnified(GravitationalTheory):
    """
    <summary>Einstein's deathbed-inspired UFT: Asymmetric metric with torsion for emergent EM, log correction for quantum bridge. g_tt = -(1 - rs/r + α log(1 + rs/r)), g_rr = 1/(1 - rs/r - α (rs/r)^2), g_φφ = r^2, g_tφ = α rs / r (torsion-like off-diagonal for EM).</summary>
    """
    category = "quantum"
    sweep = dict(alpha=np.linspace(0.007, 0.008, 5))  # Sweep around 1/137 ≈0.0073 for fine-structure coupling.

    def __init__(self, alpha: float = 1/137):
        super().__init__(f"Deathbed Unified (α={alpha:.4f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
    # <reason>α=1/137 from notes' coupling; log for "latent bridge."</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        u_g = rs / r
        log_mod = self.alpha * torch.log1p(u_g)  # Quantum instinct compression
        torsion_em = self.alpha * u_g  # Asymmetric off-diagonal for EM field
        m_sym = 1 - u_g + log_mod  # Symmetric gravity + quantum
        m_asym = -self.alpha * u_g**2  # Antisymmetric correction
        g_tt = - (m_sym + m_asym)
        g_rr = 1 / (m_sym - m_asym + EPSILON)
        g_pp = r**2
        g_tp = torsion_em * r  # Torsion-induced EM without Q
        return g_tt, g_rr, g_pp, g_tp
    # <reason>Torsion g_tp emerges EM (notes' S_μν^λ); log_mod bridges to quantum latent space.</reason>
