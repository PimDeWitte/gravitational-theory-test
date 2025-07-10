class KaluzaKleinInspiredAlpha0_5(GravitationalTheory):
    # <summary>A unified field theory inspired by Kaluza-Klein, geometrizing electromagnetism via compact extra dimensions. The metric encodes this as a dilation in g_φφ (scalar field effect) and off-diagonal g_tφ (vector potential), acting like an autoencoder compressing high-dimensional information with residual terms over radial scales. Reduces to GR at alpha=0. Key metric: g_tt = -(1 - rs/r), g_rr = 1/(1 - rs/r), g_φφ = r^2 * (1 + alpha * (rs / r)^2), g_tφ = alpha * (rs^2 / r)</summary>

    def __init__(self):
        super().__init__("KaluzaKleinInspiredAlpha0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = 0.5
        rs = 2 * G_param * M_param / (C_param ** 2)
        # <reason>Compute Schwarzschild radius rs as the fundamental geometric scale for gravity, inspired by Einstein's GR, serving as the base 'encoder' for mass information into curvature.</reason>
        g_tt = -(1 - rs / r)
        # <reason>g_tt is the standard Schwarzschild temporal component, providing the gravitational potential; kept unmodified to preserve GR limit at alpha=0, with unification effects in other components like DL residuals.</reason>
        g_rr = 1 / (1 - rs / r)
        # <reason>g_rr mirrors the inverse of -g_tt as in GR, maintaining isometric structure for radial coordinates; unification introduced elsewhere to avoid altering core gravitational encoding.</reason>
        g_phiphi = r**2 * (1 + alpha * (rs / r)**2)
        # <reason>Angular component dilated by a quadratic term mimicking Kaluza-Klein scalar field from extra dimension compaction, encoding EM repulsion geometrically as a higher-order correction, akin to a residual connection enhancing informational capacity at small radii.</reason>
        g_tphi = alpha * (rs**2 / r)
        # <reason>Off-diagonal term represents the Kaluza-Klein vector potential for electromagnetic effects, with 1/r fall-off inspired by Coulomb potential; acts as an 'attention' mechanism coupling time and angular directions, compressing quantum field info into classical geometry.</reason>
        return g_tt, g_rr, g_phiphi, g_tphi