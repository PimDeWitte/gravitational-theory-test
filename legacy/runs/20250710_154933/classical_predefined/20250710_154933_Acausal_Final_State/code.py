class AcausalFinalState(GravitationalTheory):
    """
    An acausal model considering the final state in metric calculation.
    <reason>Re-introduced from paper (Section 4.2) to test catastrophic failures in geodesic tests, as it showed high losses despite static test performance.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self):
        super().__init__("Acausal (Final State)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Simplified acausal adjustment; in practice, would require full trajectory knowledge, but approximate as perturbation
        perturbation = 0.01 * (rs / r)**2  # Placeholder for final-state influence
        m = 1 - rs / (r + EPSILON) + perturbation
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
