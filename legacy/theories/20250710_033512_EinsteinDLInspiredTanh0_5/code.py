class EinsteinDLInspiredTanh0_5(GravitationalTheory):
    """
    <summary>EinsteinDLInspiredTanh0_5: A unified field theory variant inspired by Einstein's geometric approaches (e.g., Kaluza-Klein, non-symmetric metrics) and deep learning autoencoders, treating spacetime as a compressor of high-dimensional quantum information. Introduces a non-linear geometric term alpha*(rs/r)^2 * tanh(rs/r) with alpha=0.5 as a repulsive correction mimicking electromagnetism, where tanh acts as an activation function for scale-dependent information encoding (residual-like at large rs/r, suppressed at small). Adds off-diagonal g_tφ = alpha*(rs/r)*(1 - tanh(rs/r)) for vector potential-like effects, enabling teleparallelism-inspired torsion or attention over angular coordinates. Reduces to GR at alpha=0. Key metric: g_tt = -(1 - rs/r + alpha*(rs/r)^2 * tanh(rs/r)), g_rr = 1/(1 - rs/r + alpha*(rs/r)^2 * tanh(rs/r)), g_φφ = r^2, g_tφ = alpha*(rs/r)*(1 - tanh(rs/r)).</summary>
    """

    def __init__(self):
        super().__init__("EinsteinDLInspiredTanh0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = 0.5
        rs = 2 * G_param * M_param / (C_param ** 2)
        
        # <reason>Compute Schwarzschild-like term (1 - rs/r) as the base geometric encoding of mass-energy, inspired by GR's curvature from information density.</reason>
        schwarz = 1 - rs / r
        
        # <reason>Add non-linear repulsive term alpha*(rs/r)^2 * torch.tanh(rs/r) to mimic electromagnetic repulsion geometrically, with tanh as DL activation for compressing high-dimensional quantum effects into low-dimensional spacetime, acting like a residual connection that activates strongly near the horizon (large rs/r).</reason>
        repulse = alpha * (rs / r)**2 * torch.tanh(rs / r)
        
        g_tt_factor = schwarz + repulse
        
        # <reason>g_tt incorporates the repulsive term positively to reduce gravitational attraction, encoding unified gravity-EM as geometric information bottleneck, reducing to GR at alpha=0 or large r.</reason>
        g_tt = - g_tt_factor
        
        # <reason>g_rr as inverse of g_tt_factor ensures metric consistency, inspired by Einstein's attempts to derive EM from non-symmetric or extra-dimensional geometry.</reason>
        g_rr = 1 / g_tt_factor
        
        # <reason>g_φφ remains r^2 as the base angular part, preserving spherical symmetry while allowing off-diagonal terms to introduce field-like asymmetries.</reason>
        g_phiphi = r**2
        
        # <reason>Off-diagonal g_tφ introduces a scale-dependent interaction alpha*(rs/r)*(1 - torch.tanh(rs/r)), akin to Kaluza-Klein vector potential or teleparallel torsion, with (1 - tanh) suppressing effects near horizon and enabling at larger scales, like attention mechanism over radial information flow.</reason>
        g_tphi = alpha * (rs / r) * (1 - torch.tanh(rs / r))
        
        return g_tt, g_rr, g_phiphi, g_tphi