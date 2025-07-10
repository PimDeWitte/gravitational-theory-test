class StochasticNoise(GravitationalTheory):
    """
    Tests informational robustness by adding Gaussian noise to the metric, simulating quantum fluctuations.
    <reason>Directly implements paper's recommendation (Section 3.1, 4.3.2) for noise resilience; loss measures stability as attractor. Re-introduced as a promising model for testing quantum foam hypotheses.</reason>
    """
    category = "quantum"
    sweep = None
    cacheable = True

    def __init__(self, strength: float = STOCHASTIC_STRENGTH):
        super().__init__(f"Stochastic Noise (σ={strength:.1e})")
        self.strength = torch.as_tensor(strength, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / (r + EPSILON)
        noise = torch.normal(0, self.strength, size=m.shape, device=device, dtype=DTYPE)
        m_noisy = m + noise  # Apply to g_tt; could extend to others
        return -m_noisy, 1 / (m_noisy + EPSILON), r**2, torch.zeros_like(r)
