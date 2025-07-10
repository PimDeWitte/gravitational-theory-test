class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")
        self.alpha = 0.5

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / (C_param ** 2)
        # <reason>Base factor from GR's Schwarzschild solution, representing pure geometric gravity as Einstein pursued.</reason>
        A = 1 - rs / r
        # <reason>Add parameterized geometric term alpha * (rs / r)^2 to introduce EM-like repulsive effects without explicit charge, inspired by Kaluza-Klein extra dimensions encoding EM geometrically; alpha=0 reduces to GR, non-zero adds 'residual' higher-order correction akin to DL autoencoder for compressing quantum info into classical geometry.</reason>
        A = A + self.alpha * (rs / r) ** 2
        # <reason>Include a logarithmic correction scaled by alpha, motivated by quantum scale-invariance and DL attention mechanisms over radial scales (log for multi-resolution encoding), mimicking higher-dimensional unification attempts like Einstein's.</reason>
        A = A + self.alpha * (rs / r) * torch.log(1 + r / rs)
        # <reason>g_tt as -c^2 * A, modified to include unified geometric terms for gravity-EM synthesis.</reason>
        g_tt = - (C_param ** 2) * A
        # <reason>g_rr as 1/A, preserving the inverse relation from GR but with unified modifications.</reason>
        g_rr = 1 / A
        # <reason>g_φφ as r^2, unchanged to maintain spherical symmetry in this ansatz, focusing unification on tt and rr components.</reason>
        g_φφ = r ** 2
        # <reason>Non-diagonal g_tφ introduces field-like effects via geometric off-diagonal term, inspired by Einstein's non-symmetric metric for unifying gravity and EM (antisymmetric part ~ field strength); form alpha * (rs^2 / r) acts as 'cross-attention' between time and angular, reducing to zero at alpha=0.</reason>
        g_tφ = self.alpha * (rs ** 2 / r)
        return g_tt, g_rr, g_φφ, g_tφ