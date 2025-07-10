class EinsteinFinalAlpha(GravitationalTheory):
    # <summary>Einstein-inspired unified field theory variant with parameterized geometric modification: introduces an alpha-dependent repulsive term in g_tt and g_rr, mimicking electromagnetic effects via a higher-order (rs/r)^2 correction as a 'residual connection' in the metric compression. Reduces to GR at alpha=0. Key metric: g_tt = -(1 - rs/r + alpha * (rs/r)^2), g_rr = 1/(1 - rs/r + alpha * (rs/r)^2), g_φφ = r^2 * (1 + alpha * (rs/r)^2), g_tφ = alpha * (rs/r) * torch.sin(r/rs).</summary>

    def __init__(self):
        super().__init__("EinsteinFinalAlpha")

        # <reason>Fixed alpha=0.5 for this variant, inspired by Einstein's parameterization in unified attempts; allows sweeping to test EM-like repulsion emergence from geometry, akin to Kaluza-Klein compactification scaling.</reason>
        self.alpha = 0.5

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # <reason>Compute rs = 2 * G * M / c^2, the Schwarzschild radius, as the base geometric scale; inspired by GR as the 'encoder' of mass information into curvature.</reason>
        rs = 2 * G_param * M_param / (C_param ** 2)

        # <reason>g_tt includes GR term -(1 - rs/r) plus alpha * (rs/r)^2 as a geometric correction mimicking EM repulsion, viewed as a 'residual' higher-order term in an autoencoder-like metric compression of quantum information.</reason>
        g_tt = -(1 - rs / r + self.alpha * torch.pow(rs / r, 2))

        # <reason>g_rr is inverse of the modified radial factor, maintaining metric consistency; the alpha term introduces scale-dependent 'attention' over radial distances, inspired by deep learning architectures.</reason>
        g_rr = 1 / (1 - rs / r + self.alpha * torch.pow(rs / r, 2))

        # <reason>g_φφ modified with (1 + alpha * (rs/r)^2) to introduce angular distortion, akin to extra-dimensional effects in Kaluza-Klein, encoding angular momentum information geometrically.</reason>
        g_φφ = r ** 2 * (1 + self.alpha * torch.pow(rs / r, 2))

        # <reason>Non-zero g_tφ = alpha * (rs/r) * sin(r/rs) introduces off-diagonal term for time-angular coupling, inspired by Einstein's non-symmetric metric attempts to unify EM, acting as a 'field-like' geometric twist without explicit charges.</reason>
        g_tφ = self.alpha * (rs / r) * torch.sin(r / rs)

        return g_tt, g_rr, g_φφ, g_tφ