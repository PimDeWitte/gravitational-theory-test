class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    """
    <summary>A theory drawing from Einstein's unified field theory pursuits and Kaluza-Klein ideas, introducing a parameterized geometric correction with alpha=0.5 that adds a repulsive term like (rs/r)^2 to mimic electromagnetic effects diagonally, and a non-diagonal g_tφ term inspired by extra-dimensional vector potentials acting as residual connections in a DL-like compression framework. Reduces to GR at alpha=0. Key metric: g_tt = -(1 - rs/r + alpha*(rs/r)^2), g_rr = 1/(1 - rs/r + alpha*(rs/r)^2), g_φφ = r^2, g_tφ = alpha*(rs / r)</summary>
    """

    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")
        # <reason>Set alpha=0.5 as a fixed parameter for this variant, allowing sweeps in future subclasses; inspired by Einstein's parameterized modifications to geometry in unified theories, where non-zero alpha introduces EM-like effects emerging from pure geometry, akin to Kaluza-Klein's extra dimensions compressing information into 4D fields.</reason>
        self.alpha = 0.5

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # <reason>Compute rs = 2GM/c^2, the Schwarzschild radius, as the base geometric scale; this is the standard GR starting point, viewed as the primary compression of mass-energy information into spacetime curvature.</reason>
        rs = 2 * G_param * M_param / (C_param ** 2)
        
        # <reason>Define the core factor f = 1 - rs/r + alpha*(rs/r)^2; inspired by Reissner-Nordström's charge term, but geometrically derived without explicit Q, treating the quadratic as a higher-order residual correction in an autoencoder-like metric that encodes quantum/EM information into classical geometry; alpha parameterizes the strength of this 'unified' term, reducing to GR at alpha=0.</reason>
        f = 1 - (rs / r) + self.alpha * (rs / r) ** 2
        
        # <reason>Set g_tt = -f; the negative sign ensures attractive gravity base, with the alpha term adding repulsion like EM, conceptualizing it as decompressing high-dimensional information via geometric expansion at small r.</reason>
        g_tt = -f
        
        # <reason>Set g_rr = 1/f; reciprocal to maintain metric consistency in radial direction, akin to an inverse decoding layer in DL architectures.</reason>
        g_rr = 1 / f
        
        # <reason>Set g_φφ = r^2; standard angular component, unchanged as the theory focuses on radial unification without altering transverse geometry.</reason>
        g_φφ = r ** 2
        
        # <reason>Set g_tφ = alpha * (rs / r); introduces a non-diagonal term inspired by Kaluza-Klein's off-diagonal components representing electromagnetic vector potentials, acting as an 'attention' mechanism over angular scales in the DL analogy, potentially mimicking magnetic effects without explicit fields; scales as 1/r for field-like behavior.</reason>
        g_tφ = self.alpha * (rs / r)
        
        return g_tt, g_rr, g_φφ, g_tφ