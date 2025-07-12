class NewtonianLimit(GravitationalTheory):
    """
    The Newtonian approximation of gravity.
    <reason>This theory is included as a 'distinguishable' model. It correctly lacks spatial curvature (g_rr = 1), and its significant but finite loss value validates the framework's ability to quantify physical incompleteness.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self):
        super().__init__("Newtonian Limit")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        return -m, torch.ones_like(r), r**2, torch.zeros_like(r)

class EinsteinRegularized(GravitationalTheory):
    """
    A regularized version of GR that avoids a central singularity.
    <reason>This model is a key 'distinguishable' theory. It modifies GR only at the Planck scale, and its tiny but non-zero loss demonstrates the framework's extreme sensitivity to subtle physical deviations.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self):
        super().__init__("Einstein Regularised Core")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / torch.sqrt(r**2 + LP**2)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class VariableG(GravitationalTheory):
    """
    A model where the gravitational constant G varies with distance.
    <reason>This theory tests the fundamental assumption of a constant G. The asymmetric failure (stable for weakening G, unstable for strengthening G) provides a powerful insight into the necessary conditions for a stable universe.</reason>
    """
    category = "classical"
    sweep = dict(delta=np.linspace(-0.5, 0.1, 7))
    cacheable = True

    def __init__(self, delta: float):
        super().__init__(f"Variable G (δ={delta:+.2f})")
        self.delta = torch.as_tensor(delta, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        G_eff = G_param * (1 + self.delta * torch.log1p(r / rs))
        m = 1 - 2 * G_eff * M_param / (C_param**2 * r)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class LinearSignalLoss(GravitationalTheory):
    """
    Introduces a parameter that smoothly degrades the gravitational signal as a function of proximity to the central mass.
    <reason>Re-introduced from paper (Section 3.1) as a promising model to measure breaking points in informational fidelity, analogous to compression quality degradation.</reason>
    """
    category = "classical"
    sweep = dict(gamma=np.linspace(0.0, 1.0, 5))
    cacheable = True

    def __init__(self, gamma: float):
        super().__init__(f"Linear Signal Loss (γ={gamma:+.2f})")
        self.gamma = torch.as_tensor(gamma, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        m = (1 - degradation) * (1 - rs / (r + EPSILON))
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class Participatory(GravitationalTheory):
    """
    A model where the metric is a weighted average of GR and flat spacetime, simulating observer participation.
    <reason>Re-introduced from paper (Section 4.3.1) as it demonstrates geometric brittleness; small deviations cause rapid degradation, highlighting GR's precision.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self, weight: float = 0.92):
        super().__init__(f"Participatory (w={weight:.2f})")
        self.weight = torch.as_tensor(weight, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m_gr = 1 - rs / (r + EPSILON)
        m_flat = torch.ones_like(r)
        m = self.weight * m_gr + (1 - self.weight) * m_flat
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

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

class EmergentHydrodynamic(GravitationalTheory):
    """
    An emergent model treating gravity as hydrodynamic flow.
    <reason>Re-introduced from paper (Section 4.2) for its high loss in dynamics, validating the framework's sensitivity to incorrect geometries.</reason>
    """
    category = "classical"
    sweep = None
    cacheable = True

    def __init__(self):
        super().__init__("Emergent (Hydrodynamic)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Hydrodynamic approximation: velocity-like term
        flow_term = 0.05 * torch.sqrt(rs / r)
        m = 1 - rs / r - flow_term
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

# --- QUANTUM THEORIES ---

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

class LogCorrected(GravitationalTheory):
    """
    A quantum gravity inspired model with a logarithmic correction term.
    <reason>This model is a high-performing 'distinguishable'. Logarithmic corrections are predicted by some quantum loop gravity theories, making this a promising candidate for a first-order quantum correction to GR.</reason>
    """
    category = "quantum"
    sweep = dict(beta=np.linspace(-0.50, 0.17, 7))
    cacheable = True

    def __init__(self, beta: float):
        super().__init__(f"Log Corrected (β={beta:+.2f})")
        self.beta = torch.as_tensor(beta, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        sr = torch.maximum(r, rs * 1.001)
        log_corr = self.beta * (rs / sr) * torch.log(sr / rs)
        m = 1 - rs / r + log_corr
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class QuantumCorrected(GravitationalTheory):
    """
    A generic model with a cubic correction term, representing some quantum effects.
    <reason>This model serves as a simple test case for higher-order corrections to the GR metric. Its performance relative to other theories helps classify the nature of potential quantum gravitational effects.</reason>
    """
    category = "quantum"
    sweep = dict(alpha=np.linspace(-2.0, 2.0, 9))
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Quantum Corrected (α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + self.alpha * (rs / r) ** 3
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

# --- EINSTEIN FINAL FAMILY (QUANTUM/UNIFIED) ---

class EinsteinFinalBase(GravitationalTheory):
    """Base class for the 'Einstein Final' model variants to standardize naming."""
    category = "quantum"
    sweep = None
    cacheable = True

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

class EinsteinFinalCubic(EinsteinFinalBase):
    """Original model: A simple cubic correction term added to the metric. (α=0 is GR)."""
    sweep = dict(alpha=np.linspace(-1.0, 1.0, 5))
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Cubic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**3
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalQuadratic(EinsteinFinalBase):
    """A quadratic correction term, testing a different power-law deviation."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Quadratic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**2
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalExponential(EinsteinFinalBase):
    """An exponentially suppressed correction, mimicking a short-range field."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Exponential, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r) * (1 - self.alpha * torch.exp(-r/rs))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalAsymmetric(EinsteinFinalBase):
    """Simulates an asymmetric metric by modifying g_tt and g_rr differently."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Asymmetric, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        mod = self.alpha * (rs/r)**2
        g_tt = -(1 - rs/r + mod)
        g_rr = 1 / (1 - rs/r - mod + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)

class EinsteinFinalTorsional(EinsteinFinalBase):
    """A quartic correction term, as a toy model for torsional effects."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Torsional, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**4
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalUnifiedAdditive(EinsteinFinalBase):
    """A test of unified theory where the EM field is added with a variable coupling."""
    category = "unified"
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Unified Additive, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        m = 1 - rs/r + self.alpha * (rq_sq / r**2)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalUnifiedMultiplicative(EinsteinFinalBase):
    """A non-linear interaction between the gravitational and EM fields."""
    category = "unified"
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Unified Multiplicative, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        m = (1 - rs/r) * (1 + self.alpha * (rq_sq / r**2))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalLogGravity(EinsteinFinalBase):
    """A logarithmic modification to the gravitational potential."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Log Gravity, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r) * (1 - self.alpha * torch.log1p(rs/r))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalResonant(EinsteinFinalBase):
    """A speculative resonant term causing oscillatory corrections."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Resonant, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**3 * torch.sin(r/rs)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalPionic(EinsteinFinalBase):
    """A Yukawa-like interaction inspired by meson physics."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Pionic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r) * torch.exp(-r / (3*rs))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalDynamicLambda(EinsteinFinalBase):
    """A 'local' cosmological constant that depends on gravitational field strength."""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Dynamic Lambda, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r - self.alpha * (rs/r)**2
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalEntropic(EinsteinFinalBase):
    r"""Models gravity as an entropic force, modifying the potential with a logarithmic term.
    <reason>Inspired by theories of emergent gravity (e.g., Verlinde), this model modifies the gravitational potential based on thermodynamic and holographic principles, where gravity arises from information entropy.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Entropic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * LP**2 / r**2 * torch.log(r / LP)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalMembrane(EinsteinFinalBase):
    r"""A correction inspired by higher-dimensional brane-world scenarios.
    <reason>In some string theory models, our universe is a 'brane' in a higher-dimensional space. This can lead to gravity 'leaking' into other dimensions, which is modeled here as a steep correction to the potential.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Membrane, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - torch.sqrt((rs/r)**2 + self.alpha * (LP/r)**4)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalGaussBonnet(EinsteinFinalBase):
    r"""A simplified model inspired by Gauss-Bonnet gravity, a common extension to GR.
    <reason>Gauss-Bonnet gravity adds a specific quadratic curvature term to the action. This phenomenological model captures the essence of such a modification with a steep 1/r^5 term that can arise in the metric solution.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Gauss-Bonnet, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**5
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalNonCommutative(EinsteinFinalBase):
    r"""A model motivated by non-commutative geometry, which regularizes the singularity.
    <reason>Non-commutative geometry suggests that spacetime coordinates do not commute at the Planck scale, which effectively 'smears' the singularity. This is modeled by an exponential term that smooths the metric core.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Non-Commutative, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs * torch.exp(-self.alpha * LP**2 / r**2)) / r
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalVacuum(EinsteinFinalBase):
    r"""A model where gravity's strength is coupled to the vacuum energy.
    <reason>This tests the idea that the strength of gravity could be affected by the energy density of the quantum vacuum, modeled here as a constant offset to the metric potential controlled by alpha.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Vacuum Coupling, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (LP/rs)**2
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalPowerLaw(EinsteinFinalBase):
    r"""Generalizes the potential with a variable power law, deviating from 1/r.
    <reason>This is a fundamental test of the inverse-square law at relativistic scales. By allowing the exponent to deviate from 1 (via alpha), we can test for large-scale modifications to gravity.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Power Law, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r)**(1.0 - self.alpha)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalConformal(EinsteinFinalBase):
    r"""A model inspired by conformal gravity, where physics is invariant under scale transformations.
    <reason>Conformal gravity is an alternative to GR that has different properties at cosmological scales. This model introduces a term that respects conformal symmetry, testing a different geometric foundation.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Conformal, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * r
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalDilaton(EinsteinFinalBase):
    r"""A model including a dilaton field from string theory.
    <reason>String theory predicts the existence of a scalar field, the dilaton, which couples to gravity. This model tests a simple form of this coupling, modifying the strength of the gravitational potential.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Dilaton, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r) / (1 + self.alpha * (rs/r))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalTachyonic(EinsteinFinalBase):
    r"""A speculative model with a tachyonic field contribution.
    <reason>Tachyonic fields, while problematic, appear in some string theory contexts. This model tests the effect of a potential that weakens at short distances, a hallmark of tachyon condensation.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Tachyonic, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r * (1 - self.alpha * torch.tanh(rs/r))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalHigherDeriv(EinsteinFinalBase):
    r"""A model with both quadratic and cubic corrections.
    <reason>Instead of testing just one higher-order term, this model includes two, allowing for more complex interactions and a better fit if the true quantum corrections are not simple power laws.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Higher-Derivative, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**2 - self.alpha * (rs/r)**3
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalQuintessence(EinsteinFinalBase):
    r"""A model that includes a quintessence-like scalar field.
    <reason>Quintessence is a hypothesized form of dark energy. This models its effect on local spacetime geometry as a very shallow power-law term, distinct from a cosmological constant.</reason>"""
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Final (Quintessence, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r - self.alpha * (r/rs)**0.5
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinUnifiedGeometricField(EinsteinFinalBase):
    r"""A candidate for Einstein's final theory, synthesizing his work on unification.
    <reason>This theory represents a culmination of the project's goals. It is a creative, physically motivated attempt to model the principles of unification that Einstein pursued. It combines his known theoretical approaches into a single, testable hypothesis.</reason>"""
    category = "unified"
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Einstein Unified Geometric Field, α={alpha:+.2f}")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        u_g = rs / r
        u_e = rq_sq / r**2
        log_mod = self.alpha * torch.log1p(u_g)
        unified_potential = u_g - (u_e / (1 + u_g))
        g_tt = -(1 - unified_potential + log_mod)
        g_rr = 1 / (1 - unified_potential - log_mod + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)

class EinsteinUnifiedGeometricField2(GravitationalTheory):
    """
    A candidate for Einstein's final theory, synthesizing his work on unification.

    This model attempts to unify gravity and electromagnetism through a purely
    geometric framework, inspired by three key principles from Einstein's later work:

    1.  **Asymmetric Metric**: The metric's time and space components are modified
        differently, a phenomenological approach to an asymmetric metric tensor
        ($g_{μν} ≠ g_{νμ}$), where the antisymmetric part was hoped to
        describe electromagnetism.

    2.  **Geometric Source for Electromagnetism**: The electromagnetic term is not
        added, but arises from a non-linear interaction between the gravitational
        potential ($r_s/r$) and the charge potential ($r_q^2/r^2$). This models the
        idea that the electromagnetic field is a feature of the gravitational field,
        not separate from it.

    3.  **Logarithmic Potential**: A logarithmic term is included, representing a
        subtle, long-range modification to the geometry. This can be interpreted as
        a nod to the need for a deeper theory underlying quantum mechanics, introducing
        a new informational layer or "hidden variable" into the geometry itself,
        consistent with Einstein's desire for a more complete, deterministic reality.

    <reason>This theory represents a culmination of the project's goals. It is a creative, physically motivated attempt to model the principles of unification that Einstein pursued. It combines his known theoretical approaches into a single, testable hypothesis. Its performance against the dual baselines will be the ultimate test of this information-theoretic framework.</reason>
    """
    category = "unified"
    sweep = None
    cacheable = True

    def __init__(self, alpha: float):
        super().__init__(f"Unified Geometric Field 2 (α={alpha:+.3f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        u_g = rs / r  # Gravitational potential
        u_e = rq_sq / r**2 # Electromagnetic potential
        log_mod = self.alpha * torch.log1p(u_g)
        unified_potential = u_g - (u_e / (1 + u_g))
        g_tt = -(1 - unified_potential + log_mod)
        g_rr = 1 / (1 - unified_potential - log_mod + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)

class EinsteinFinalUnifiedTheory(GravitationalTheory):
    r"""The culmination of Einstein's quest for a unified field theory.
    <reason>This model is the most ambitious synthesis, combining a non-linear reciprocal coupling of gravity and EM with a hyperbolic term representing a deterministic substructure. Its success would be the strongest possible validation of the paper's thesis.</reason>"""
    category = "unified"
    sweep = None
    cacheable = True

    def __init__(self, gamma: float):
        super().__init__(f"Einstein's UFT (γ={gamma:+.3f})")
        self.gamma = torch.as_tensor(gamma, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        u_g = rs / r
        u_e = rq_sq / r**2
        hyp_mod = self.gamma * torch.cosh(u_e / (u_g + EPSILON)) - self.gamma
        unified_potential = u_g / (1 + u_g * u_e) + u_e / (1 + u_e / u_g + EPSILON)
        g_tt = -(1 - unified_potential + hyp_mod / 2)
        g_rr = 1 / (1 - unified_potential - hyp_mod + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)

class EinsteinDeathbedUnified(GravitationalTheory):
    r"""
    <summary>Einstein's deathbed-inspired UFT: Asymmetric metric with torsion for emergent EM, log correction for quantum bridge. g_tt = -(1 - rs/r + α log(1 + rs/r)), g_rr = 1/(1 - rs/r - α (rs/r)^2), g_φφ = r^2, g_tφ = α rs / r (torsion-like off-diagonal for EM).</summary>
    """
    category = "unified"
    sweep = dict(alpha=np.linspace(0.007, 0.008, 5))  # Sweep around 1/137 ≈0.0073 for fine-structure coupling.
    cacheable = True

    def __init__(self, alpha: float = 1/137):
        super().__init__(f"Deathbed Unified (α={alpha:.4f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
    # <reason>α=1/137 from notes' coupling; log for "latent bridge."</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        u_g = rs / r
        log_mod = self.alpha * torch.log1p(u_g)  # Quantum instinct compression
        torsion_em = self.alpha * u_g  # Asymmetric off-diagonal for EM field
        m_sym = 1 - u_g + log_mod  # Symmetric gravity + quantum
        m_asym = -self.alpha * u_g**2  # Antisymmetric correction
        g_tt = - (m_sym + m_asym)
        g_rr = 1 / (m_sym - m_asym + EPSILON)
        g_pp = r**2
        g_tp = torsion_em * r  # Torsion-induced EM without Q
        return g_tt, g_rr, g_pp, g_tp
    # <reason>Torsion g_tp emerges EM (notes' S_μν^λ); log_mod bridges to quantum latent space.</reason>

class SignalLossWrapper(GravitationalTheory):
    """
    Wrapper to add linear signal loss degradation to any base theory.
    <reason>Enables efficient introduction of signal loss to multiple Einstein final theories for analysis, as per user request.</reason>
    """
    category = "wrapped"
    sweep = None
    cacheable = True

    def __init__(self, base_theory: GravitationalTheory, gamma: float):
        super().__init__(f"{base_theory.name} with Signal Loss (γ={gamma:+.2f})")
        self.base_theory = base_theory
        self.gamma = torch.as_tensor(gamma, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        g_tt, g_rr, g_pp, g_tp = self.base_theory.get_metric(r, M_param, C_param, G_param)
        rs = 2 * G_param * M_param / C_param**2
        degradation = self.gamma * (rs / r)
        # Assume m = -g_tt (time-like component)
        base_m = -g_tt
        degraded_m = (1 - degradation) * base_m
        degraded_g_tt = -degraded_m
        # Adjust g_rr consistently, assuming isotropic form
        degraded_g_rr = 1 / (degraded_m + EPSILON)
        return degraded_g_tt, degraded_g_rr, g_pp, g_tp

class QuantumLinearSignalLoss(GravitationalTheory):
    """
    Combines logarithmic quantum correction with linear signal loss.
    <reason>Extends LinearSignalLoss with quantum log term to test combined effects, based on promising LogCorrected performance.</reason>
    """
    category = "quantum"
    sweep = dict(beta=np.linspace(-0.5, 0.5, 5), gamma=np.linspace(0.0, 1.0, 5))
    cacheable = True

    def __init__(self, beta: float, gamma: float):
        super().__init__(f"Quantum Linear Signal Loss (β={beta:+.2f}, γ={gamma:+.2f})")
        self.beta = torch.as_tensor(beta, device=device, dtype=DTYPE)
        self.gamma = torch.as_tensor(gamma, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        sr = torch.maximum(r, rs * 1.001)
        log_corr = self.beta * (rs / sr) * torch.log(sr / rs)
        base_m = 1 - rs / r + log_corr
        degradation = self.gamma * (rs / r)
        degraded_m = (1 - degradation) * base_m
        g_tt = -degraded_m
        g_rr = 1 / (degraded_m + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)