class KaluzaKleinGeometric(GravitationalTheory):
    # <summary>Kaluza-Klein inspired theory unifying gravity and electromagnetism via an extra compact dimension, with C_param as the compactification scale introducing geometric modifications without explicit charge. Key metric: g_tt = -(1 - rs/r) * (1 + (C/r)^2), g_rr = 1/(1 - rs/r) / (1 + (C/r)^2), g_φφ = r^2 * (1 + (C/r)^2), g_tφ = (C/r) * (rs/r)</summary>

    def __init__(self):
        super().__init__("KaluzaKleinGeometric")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # <reason>Compute Schwarzschild radius rs using G_param and M_param, inspired by GR foundation, to serve as the base for geometric unification.</reason>
        rs = 2 * G_param * M_param

        # <reason>Introduce compactification term (C/r)^2 geometrically, mimicking the projection from 5D to 4D in Kaluza-Klein, where extra dimension encodes EM-like effects as spacetime compression, akin to autoencoder dimensionality reduction of quantum information.</reason>
        compact_term = (C_param / r) ** 2

        # <reason>Modify g_tt by multiplying the Schwarzschild factor with (1 + compact_term) to represent the dilaton-like scaling from KK reduction, viewing it as encoding high-dimensional information into the time component.</reason>
        g_tt = - (1 - rs / r) * (1 + compact_term)

        # <reason>Set g_rr as inverse of Schwarzschild-like factor divided by (1 + compact_term), inspired by conformal transformations in KK theories, acting as a decoder for radial geometry in the information compression analogy.</reason>
        g_rr = 1 / ((1 - rs / r) * (1 + compact_term))

        # <reason>Scale g_φφ by (1 + compact_term) to reflect the extra dimension's contribution to angular geometry, geometrically unifying with gravity, and representing decompression of angular degrees of freedom in the autoencoder-inspired framework.</reason>
        g_phiphi = r ** 2 * (1 + compact_term)

        # <reason>Add off-diagonal g_tφ proportional to (C/r) * (rs/r) to mimic the vector potential A_φ from KK compactification, introducing a geometric 'twist' that encodes electromagnetic information off-diagonally, testable for fidelity in decoding loss.</reason>
        g_tphi = (C_param / r) * (rs / r)

        return g_tt, g_rr, g_phiphi, g_tphi