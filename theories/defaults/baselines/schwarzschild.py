from base_theory import GravitationalTheory, Tensor
import torch

class Schwarzschild(GravitationalTheory):
    """
    The Schwarzschild metric - exact solution to Einstein's field equations for a non-rotating, uncharged mass.
    This is our baseline for pure gravity.
    """
    category = "classical"
    cacheable = True

    def __init__(self):
        super().__init__("Schwarzschild (GR)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        return -m, 1 / (m + 1e-10), r**2, torch.zeros_like(r) 