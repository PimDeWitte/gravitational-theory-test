```python
class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    # <summary>A parameterized theory inspired by Einstein's unified field attempts and Kaluza-Klein, introducing geometric EM-like effects via a repulsive quadratic term and off-diagonal g_tφ without explicit charge. Reduces to GR at alpha=0. Key metric: f = 1 - rs/r + alpha (rs/r)^2, g_tt = -f, g_rr = 1/f, g_φφ = r^2, g_tφ = alpha (rs^2 / r), with alpha=0.5 and rs=2GM/c^2.</summary>

    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")
        self.alpha = 0.5  # Fixed alpha=0.5 for this variant, enabling parameter sweeps in evaluations

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # <reason>Compute Schwarzschild radius rs geometrically from parameters, serving as the base scale for compression of mass information into geometry, akin to encoding mass in low-dimensional spacetime.</reason>
        rs = 2 * G_param * M_param / (C_param ** 2)
        
        # <reason>Define the radial function f with GR term (1 - rs/r)