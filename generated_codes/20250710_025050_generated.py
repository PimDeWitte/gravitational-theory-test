class EinsteinUnifiedAlpha0_5(GravitationalTheory):
    def __init__(self):
        super().__init__("EinsteinUnifiedAlpha0_5")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = C_param
        # <reason> Use C_param as alpha, a tunable parameter for sweeping over unification strength, inspired by Einstein's iterative modifications in pursuit of unified field theory, allowing reduction to GR at alpha=0 while introducing EM-like effects at non-zero values. </reason>
        rs = 2 * G_param * M_param
        # <reason> Compute Schwarzschild radius rs as base geometric scale, serving as the 'bottleneck' in the autoencoder analogy where high-dimensional information is compressed into classical geometry. </reason>
        delta = 1 - rs / r + alpha * (rs / r)**2 * torch.log1p(r / rs)
        # <reason> Modify the metric factor delta with a term alpha * (rs/r)^2 * log(1 + r/rs), inspired by Einstein's attempts to geometrize electromagnetism via higher-order or non-polynomial terms; the quadratic mimics RN charge repulsion geometrically, while log introduces quantum-inspired corrections akin to residual connections or scale-attentive layers in deep learning, encoding multi-scale information from higher dimensions. </reason>
        g_tt = -delta
        # <reason> Set g_tt = -delta to maintain time-like signature, with the modification providing repulsive effects similar to electromagnetic fields emerging from pure geometry, as in Kaluza-Klein compactification. </reason>
        g_rr = 1 / delta
        # <reason> Set g_rr = 1/delta for consistency with isotropic form, ensuring the geometric perturbation affects radial stretching in a way that could decode to charged particle behavior. </reason>
        g_phiphi = r**2
        # <reason> Keep g_φφ = r^2 as the standard angular component, preserving spherical symmetry while allowing perturbations elsewhere to unify fields without altering base topology. </reason>
        g_tphi = alpha * (rs**2 / r)
        # <reason> Introduce non-zero g_tφ = alpha * (rs^2 / r) for off-diagonal coupling, drawn from Einstein's non-symmetric metric ideas and Kaluza-Klein off-diagonal terms representing vector fields; this adds a 'teleparallel-like' twist, interpretable as attention over temporal-angular directions in the DL framework, potentially encoding magnetic or rotational EM effects geometrically. </reason>
        return g_tt, g_rr, g_phiphi, g_tphi