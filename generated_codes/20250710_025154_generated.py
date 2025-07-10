class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # <reason>Compute the Schwarzschild radius rs using standard GR formula, serving as the geometric foundation for gravity inspired by Einstein's equivalence principle and curvature-based unification attempts.</reason>
        rs = 2 * G_param * M_param / (C_param ** 2)

        # <reason>Set alpha=0.5 as a fixed parameter for this subclass, inspired by Einstein's 30-year pursuit of unified theories with adjustable constants to blend gravity and EM; enables testing how non-zero alpha introduces EM-like effects while reducing to GR at alpha=0.</reason>
        alpha = 0.5

        # <reason>Introduce a repulsive geometric term alpha * (rs**2 / r**2) to mimic the Q^2/r^2 repulsion in Reissner-Nordström without explicit charge, drawing from Kaluza-Klein extra dimensions where compactified geometry induces effective EM fields as pure geometry.</reason>
        repulsion_term = alpha * torch.pow(rs / r, 2)

        # <reason>Add a higher-order logarithmic correction inspired by quantum gravitational effects (e.g., renormalization group flows) and deep learning autoencoders, where log terms act as attention mechanisms over radial scales, compressing high-dimensional information into the metric like residual connections for better fidelity in decoding quantum states to classical spacetime.</reason>
        log_correction = alpha * (rs / r) * torch.log(1 + (rs / r))

        # <reason>Combine terms into A for g_tt and g_rr, ensuring asymptotic flatness while modifying near-field behavior to encode unified field effects, analogous to Einstein's non-symmetric metric approaches where geometry alone generates field strengths.</reason>
        A = 1 - rs / r + repulsion_term + log_correction

        g_tt = -A
        g_rr = 1 / A

        # <reason>Set g_φφ to r**2 as in standard spherical geometry, maintaining the angular part unchanged to focus unification efforts on temporal-radial modifications, consistent with Einstein's geometric unification strategies.</reason>
        g_φφ = r**2

        # <reason>Introduce non-diagonal g_tφ term proportional to alpha * rs**2 / r, inspired by Einstein's asymmetric metric theories (e.g., teleparallelism) where skew components encode electromagnetic potentials geometrically, akin to Kaluza-Klein vector fields or DL cross-attention between time and angular coordinates for enriched information encoding.</reason>
        g_tφ = -alpha * (rs**2 / r)

        return g_tt, g_rr, g_φφ, g_tφ