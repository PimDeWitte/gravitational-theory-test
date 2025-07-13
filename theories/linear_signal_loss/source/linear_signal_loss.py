
# --- CLASSICAL THEORIES ---

import torch
import math
from scipy.constants import G, c, hbar
from base_theory import GravitationalTheory, Tensor
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32
EPSILON = torch.finfo(DTYPE).eps * 100
LP = torch.as_tensor(math.sqrt(G * hbar / c**3), device=device, dtype=DTYPE)

# --- QUANTUM EXTENSION ---

class QuantumLinearSignalLoss(GravitationalTheory):
    """
    Quantum extension of Linear Signal Loss with Planck-scale correction.
    <reason>Addresses feedback on quantum scales by adding a minimal length correction, testing if unification holds near quantum regimes.</reason>
    """
    category = "quantum"
    cacheable = True

    def __init__(self, beta: float = 0.1):
        super().__init__(f"Quantum Linear Signal Loss (β={beta:+.2f})")
        self.gamma = torch.as_tensor(1.00, device=device, dtype=DTYPE)
        self.beta = torch.as_tensor(beta, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        quantum_corr = self.beta * (LP / r)**2  # Planck-scale correction
        m = (1 - degradation) * (1 - rs / (r + EPSILON)) + quantum_corr
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

