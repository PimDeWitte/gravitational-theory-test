
# --- CLASSICAL THEORIES ---

class Schwarzschild(GravitationalTheory):
    """
    The Schwarzschild metric for a non-rotating, uncharged black hole.
    <reason>This is the exact solution to Einstein's field equations in a vacuum and serves as the fundamental ground truth (baseline) for pure gravity in this framework.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self):
        super().__init__("Schwarzschild (GR)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / (r + EPSILON)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

# Create explicit LinearSignalLoss variants instead of using sweep
class LinearSignalLoss_gamma_0_00(GravitationalTheory):
    """
    Linear Signal Loss with γ=0.00
    <reason>Baseline - no signal degradation, equivalent to Schwarzschild</reason>
    """
    category = "classical"
    cacheable = True

    def __init__(self):
        super().__init__("Linear Signal Loss (γ=+0.00)")
        self.gamma = torch.as_tensor(0.00, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + EPSILON))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class LinearSignalLoss_gamma_0_25(GravitationalTheory):
    """
    Linear Signal Loss with γ=0.25
    <reason>25% signal degradation based on proximity to central mass</reason>
    """
    category = "classical"
    cacheable = True

    def __init__(self):
        super().__init__("Linear Signal Loss (γ=+0.25)")
        self.gamma = torch.as_tensor(0.25, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + EPSILON))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class LinearSignalLoss_gamma_0_50(GravitationalTheory):
    """
    Linear Signal Loss with γ=0.50
    <reason>50% signal degradation - significant compression loss</reason>
    """
    category = "classical"
    cacheable = True

    def __init__(self):
        super().__init__("Linear Signal Loss (γ=+0.50)")
        self.gamma = torch.as_tensor(0.50, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + EPSILON))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class LinearSignalLoss_gamma_0_75(GravitationalTheory):
    """
    Linear Signal Loss with γ=0.75
    <reason>75% signal degradation - severe compression artifacts</reason>
    """
    category = "classical"
    cacheable = True

    def __init__(self):
        super().__init__("Linear Signal Loss (γ=+0.75)")
        self.gamma = torch.as_tensor(0.75, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + EPSILON))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class LinearSignalLoss_gamma_1_00(GravitationalTheory):
    """
    Linear Signal Loss with γ=1.00
    <reason>100% signal degradation at Schwarzschild radius - complete information loss</reason>
    """
    category = "classical"
    cacheable = True

    def __init__(self):
        super().__init__("Linear Signal Loss (γ=+1.00)")
        self.gamma = torch.as_tensor(1.00, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + EPSILON))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
