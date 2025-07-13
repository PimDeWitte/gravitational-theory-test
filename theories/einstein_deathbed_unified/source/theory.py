from base_theory import GravitationalTheory, Tensor
import torch
import numpy as np

class EinsteinDeathbedUnified(GravitationalTheory):
    r"""
    <summary>Einstein's deathbed-inspired UFT: Asymmetric metric with torsion for emergent EM, log correction for quantum bridge. g_tt = -(1 - rs/r + α log(1 + rs/r)), g_rr = 1/(1 - rs/r - α (rs/r)^2), g_φφ = r^2, g_tφ = α rs / r (torsion-like off-diagonal for EM).</summary>
    """
    category = "unified"
    sweep = dict(alpha=np.linspace(0.007, 0.008, 5))  # Sweep around 1/137 ≈0.0073 for fine-structure coupling.
    cacheable = True

    def __init__(self, alpha: float = 1/137):
        super().__init__(f"Deathbed Unified (α={alpha:.4f})")
        # Don't force device - let the framework handle it
        self.alpha = alpha

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        u_g = rs / r
        # Ensure alpha is on same device as r
        alpha = torch.tensor(self.alpha, device=r.device, dtype=r.dtype)
        log_mod = alpha * torch.log1p(u_g)
        torsion_em = alpha * u_g
        m_sym = 1 - u_g + log_mod
        m_asym = -alpha * u_g**2
        g_tt = - (m_sym + m_asym)
        g_rr = 1 / (m_sym - m_asym + 1e-10)
        g_pp = r**2
        g_tp = torsion_em * r
        return g_tt, g_rr, g_pp, g_tp 