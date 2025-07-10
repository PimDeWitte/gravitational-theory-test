# <summary>A unified field theory inspired by Einstein's pursuit and Kaluza-Klein theory, introducing parameterized exponential dilation in g_φφ and off-diagonal g_tφ with exponential decay to encode electromagnetic-like effects geometrically. This is viewed as a deep learning-inspired architecture where the exponential terms act as gating mechanisms in the autoencoder-like compression of high-dimensional quantum information into classical spacetime, with attention over decaying radial scales. Reduces to GR at alpha=0. Key metric: g_tt = -(1 - rs/r), g_rr = 1/(1 - rs/r), g_φφ = r^2 * (1 + alpha * exp(-rs / r)), g_tφ = alpha * (rs / r) * exp(-rs / r) * r, with alpha=0.5.</summary>
class EinsteinKaluzaExpAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinKaluzaExpAlpha0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = torch.tensor(0.5, device=r.device, dtype=r.dtype)
        rs = 2 * G_param * M_param / (C_param ** 2)
        # <reason>Inspired by GR's Schwarzschild metric for the time component, kept unchanged to preserve gravitational attraction while modifications elsewhere encode EM-like effects.</reason>
        g_tt = -(1 - rs / r)
        # <reason>Standard GR radial component, unaltered to maintain compatibility with GR limit at alpha=0, focusing modifications on angular and off-diagonal terms for unified field encoding.</reason>
        g_rr = 1 / (1 - rs / r)
        # <reason>Exponential dilation in g_φφ inspired by Kaluza-Klein compact dimensions, where the exp(-rs/r) term acts as a gating function compressing high-dimensional information, with alpha parameterizing the strength of this geometric "scalar field" effect akin to DL residual scaling.</reason>
        g_φφ = r**2 * (1 + alpha * torch.exp(-rs / r))
        # <reason>Off-diagonal g_tφ with exponential decay, mimicking electromagnetic vector potential geometrically, viewed as an attention mechanism between time and angular coordinates for multi-scale information flow in the spacetime autoencoder.</reason>
        g_tφ = alpha * (rs / r) * torch.exp(-rs / r) * r
        return g_tt, g_rr, g_φφ, g_tφ