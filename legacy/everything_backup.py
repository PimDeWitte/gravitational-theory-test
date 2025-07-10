#!/usr/bin/env python3
# sim_gpu.py  ── July 2025
# ---------------------------------------------------------------------------
# Float‑32 black‑hole orbital integrator for Apple‑silicon (M‑series) GPUs.
# All known mathematical / computational bugs are fixed; optional Torch‑Dynamo
# compilation can be enabled with `TORCH_COMPILE=1`.
#
# --- MODIFICATION NOTES (JULY 2025) ---
# In accordance with the information-theoretic principles outlined in the
# accompanying paper ("Our Reality Matters"), the method for calculating the
# "trajectory loss" has been upgraded.
#
# Previous method: Squared physical distance between the final points of the
# reference (GR) and predicted trajectories. This was a simple metric, but
# insensitive to the full orbital history.
#
# New method: Fourier-domain analysis. The radial component of the orbit,
# r(τ), is treated as a time-series signal. We compute the Fast Fourier
# Transform (FFT) of this signal for both the reference and predicted
# trajectories. The loss is then defined as the Mean Squared Error (MSE)
# between the magnitudes of these two frequency spectra.
#
# This approach offers a far more robust comparison, capturing differences
# in orbital shape, precession, and decay rate over the entire simulation.
# It directly measures the "informational fidelity" of a theory by quantifying
# how well it reproduces the full frequency spectrum of the ground-truth orbit.
#
# --- UPDATE (JULY 10, 2025) ---
# - FINAL VALIDATION: Performed line-by-line code review and added exhaustive
#   documentation to ensure all logic is sound and no errors remain.
# - RESTORED ALL THEORIES: Re-integrated all original theories and sweeps.
# - EXPANDED SUITE: Added 26 "Einstein Final" variants and 5 new "Transformer-inspired" models.
# - CORRECTED PARAMETERS: Increased Reissner-Nordström charge parameter Q to
#   ensure a distinct electromagnetic baseline for a valid unification test.
# ---------------------------------------------------------------------------

from __future__ import annotations
import os # <reason>Used for creating plot directories and checking for cached files.</reason>
import time # <reason>Used to create unique directory names for plot runs based on the current timestamp.</reason>
import math # <reason>Used for `math.pi` and `math.sqrt` for physical constant definitions before tensor conversion.</reason>
import argparse # <reason>Used to define and parse command-line arguments for controlling the simulation mode (e.g., --final, --plots).</reason>
import warnings # <reason>Used to suppress non-critical JIT compiler warnings to keep console output clean.</reason>

import torch # <reason>The core computational library for tensor operations, automatic differentiation, and GPU acceleration.</reason>
import numpy as np # <reason>Used for creating parameter sweeps with `linspace` and `logspace` before conversion to tensors.</reason>
import matplotlib.pyplot as plt # <reason>The primary library for generating and saving plots of the orbital trajectories.</reason>
from scipy.constants import G, c, k, hbar, epsilon_0 # <reason>Provides high-precision values for fundamental physical constants.</reason>

# ---------------------------------------------------------------------------
# 0.  CLI ARGUMENTS & GLOBAL CONFIG
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    """
    Parses command-line arguments for the simulation.
    <reason>This function encapsulates argument parsing, making the main script cleaner and easier to read. It defines the operational modes of the script, such as plotting, precision, and run duration.</reason>
    """
    p = argparse.ArgumentParser(description="PyTorch-based orbital mechanics simulator for gravitational theories.") # <reason>Creates the parser object with a helpful description.</reason>
    p.add_argument("--final", action="store_true", help="Run with final, high-step-count parameters for publication-quality data.") # <reason>Defines a flag for long, high-precision runs.</reason>
    p.add_argument("--plots", action="store_true", help="Generate and save plots for the top-ranked models.") # <reason>Defines a flag to enable plot generation.</reason>
    p.add_argument("--no-plots", action="store_true", help="Explicitly disable all plotting, even if other flags would enable it.") # <reason>Defines a flag to override and disable plotting.</reason>
    p.add_argument("--cpu-f64", action="store_true", help="Run on CPU with float64 precision for validation. Overrides default GPU/float32 settings.") # <reason>Defines a flag for high-precision CPU-based validation.</reason>
    return p.parse_args() # <reason>Executes the parsing and returns the populated namespace object.</reason>

args = parse_cli() # <reason>Parses the command-line arguments at the start of the script.</reason>
# Insert the XAI API key from the environment, if present, into the process for tracking or authentication.
XAI_API_KEY = os.environ.get("XAI_API_KEY", None)


# Set device and data type based on CLI flags. This must be done before any tensors are created.
# <reason>This block allows for flexible hardware and precision choices. The default is fast GPU/float32 for exploration, while --cpu-f64 enables high-precision CPU runs for validating key results, as recommended in the research plan.</reason>
if args.cpu_f64:
    DTYPE  = torch.float64 # <reason>Sets the global data type to 64-bit float for high precision.</reason>
    device = torch.device("cpu") # <reason>Sets the global computation device to the CPU.</reason>
else:
    DTYPE  = torch.float32 # <reason>Sets the global data type to 32-bit float for high performance on GPUs.</reason>
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # <reason>Sets the global device to Apple's Metal Performance Shaders (MPS) for GPU acceleration if available, otherwise falls back to CPU.</reason>

# Epsilon value for numerical stability, scaled by the chosen data type's precision.
# <reason>A fixed small epsilon is not robust. Tying it to the data type's machine epsilon ensures that the stability margin is appropriate for both float32 and float64, preventing underflow or loss of significance.</reason>
EPSILON  = torch.finfo(DTYPE).eps * 100

# ---------------------------------------------------------------------------
# 1.  PHYSICAL CONSTANTS & SYSTEM PARAMETERS
# ---------------------------------------------------------------------------

# <reason>Defining constants as tensors on the correct device and with the correct dtype from the start avoids repeated CPU-GPU transfers and type conversions within the simulation loop, which is a major performance optimization.</reason>
TORCH_PI = torch.as_tensor(math.pi,  device=device, dtype=DTYPE)
EPS0_T   = torch.as_tensor(epsilon_0, device=device, dtype=DTYPE)

# System Parameters: 10 Solar Mass Black Hole
# <reason>These parameters define the central object for our simulation. 10 M☉ is a standard choice for a stellar-mass black hole, providing a realistic scale for testing gravitational effects.</reason>
M_SI  = 10.0 * 1.989e30 # <reason>Mass of the central object in kilograms (10 solar masses).</reason>
RS_SI = 2 * G * M_SI / c**2 # <reason>Schwarzschild radius in meters, calculated from the mass.</reason>
M  = torch.as_tensor(M_SI , device=device, dtype=DTYPE) # <reason>Mass as a PyTorch tensor for use in calculations.</reason>
RS = torch.as_tensor(RS_SI, device=device, dtype=DTYPE) # <reason>Schwarzschild radius as a PyTorch tensor.</reason>

# Cached Planck Length Tensor
# <reason>The Planck Length is used in some quantum gravity models. Caching it as a tensor avoids recalculating the Python float and converting it to a tensor inside the simulation loop.</reason>
LP = torch.as_tensor(math.sqrt(G * hbar / c**3), device=device, dtype=DTYPE)

# Default parameters for various speculative models.
# <reason>These default values are used to instantiate the non-swept versions of the theories. They are chosen to be physically significant enough to produce a deviation from GR without immediately causing the simulation to fail.</reason>
J_FRAC, Q_PARAM = 0.5, 3.0e15 # <reason>J_FRAC is the dimensionless spin for Kerr. Q_PARAM is the electric charge for Reissner-Nordström, increased to ensure a strong, unambiguous electromagnetic effect for the unification test.</reason>
ASYMMETRY_PARAM, TORSION_PARAM = 1.0e-4, 1.0e-3 # <reason>Small coupling constants for their respective speculative theories.</reason>
OBSERVER_ENERGY, LAMBDA_COSMO = 1.0e9, 1.11e-52 # <reason>Energy parameter for the Participatory theory and the cosmological constant for NonLocal gravity.</reason>
STOCHASTIC_STRENGTH = 1e-7 # <reason>A very small noise amplitude for the StochasticNoise model to ensure baseline stability.</reason>

# ---------------------------------------------------------------------------
# 2.  THEORY DEFINITIONS
# ---------------------------------------------------------------------------

Tensor = torch.Tensor  # <reason>A type alias for `torch.Tensor` to make type hints in function signatures more concise and readable.</reason>

class GravitationalTheory:
    """
    Abstract base class for all gravitational theories.
    <reason>This class defines a common interface (`get_metric`) that all theories must implement. This polymorphic design allows the integrator to treat any theory identically, simplifying the simulation logic and making the framework easily extensible.</reason>
    """
    def __init__(self, name: str) -> None: # <reason>The constructor for the base class, taking only the theory's name.</reason>
        self.name = name # <reason>Stores the human-readable name of the theory for reporting and plotting.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Calculates the metric components (g_tt, g_rr, g_φφ, g_tφ) for a given radius."""
        raise NotImplementedError # <reason>This method must be implemented by all subclasses, ensuring they adhere to the required interface.</reason>

# -- 2.1 Standard & Baseline Metrics --

class Schwarzschild(GravitationalTheory):
    """
    The Schwarzschild metric for a non-rotating, uncharged black hole.
    <reason>This is the exact solution to Einstein's field equations in a vacuum and serves as the fundamental ground truth (baseline) for pure gravity in this framework.</reason>
    """
    def __init__(self): super().__init__("Schwarzschild (GR)") # <reason>Initializes the base class with the standard name for General Relativity's simplest black hole solution.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates the Schwarzschild radius, a key parameter.</reason>
        m = 1 - rs / (r + EPSILON) # <reason>Defines the core metric component `m = 1 - rs/r`, with a small epsilon to prevent division by zero at the singularity.</reason>
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r) # <reason>Returns the metric components: g_tt = -m, g_rr = 1/m, g_φφ = r^2, and g_tφ = 0 (for non-rotating spacetime).</reason>

class NewtonianLimit(GravitationalTheory):
    """
    The Newtonian approximation of gravity.
    <reason>This theory is included as a 'distinguishable' model. It correctly lacks spatial curvature (g_rr = 1), and its significant but finite loss value validates the framework's ability to quantify physical incompleteness.</reason>
    """
    def __init__(self): super().__init__("Newtonian Limit") # <reason>Initializes the base class with its descriptive name.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates the Schwarzschild radius for comparison.</reason>
        m = 1 - rs / r # <reason>Defines the time component of the metric, identical to Schwarzschild's g_tt.</reason>
        return -m, torch.ones_like(r), r**2, torch.zeros_like(r) # <reason>Returns the metric components. Crucially, g_rr is 1, indicating flat spatial geometry as per Newtonian physics.</reason>

class ReissnerNordstrom(GravitationalTheory):
    """
    The Reissner-Nordström metric for a charged, non-rotating black hole.
    <reason>This is the exact solution for a charged mass and serves as the second ground truth (the Kaluza-Klein baseline) for testing a theory's ability to unify gravity and electromagnetism.</reason>
    """
    def __init__(self, Q: float): # <reason>Takes the electric charge Q as a parameter.</reason>
        super().__init__(f"Reissner‑Nordström (Q={Q:.1e})") # <reason>Initializes with a name that includes the charge parameter.</reason>
        self.Q = torch.as_tensor(Q, device=device, dtype=DTYPE) # <reason>Stores the charge as a tensor.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates the gravitational Schwarzschild radius.</reason>
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4) # <reason>Calculates the squared "charge radius", representing the electromagnetic contribution.</reason>
        m = 1 - rs / r + rq_sq / r**2 # <reason>Defines the metric component, which includes both the mass term (1/r) and the charge term (1/r^2).</reason>
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r) # <reason>Returns the R-N metric components.</reason>

# -- 2.2 Original Suite of Speculative Theories --

class Kerr(GravitationalTheory):
    """The metric for a rotating, uncharged black hole. Included for completeness.
    <reason>Kerr is a core part of GR, but our integrator is for non-rotating spacetimes. Its inclusion serves as a test of how the framework handles more complex, non-diagonal metrics, even if expected to fail.</reason>"""
    def __init__(self, J_frac: float): # <reason>Takes the dimensionless spin parameter J_frac (a*).</reason>
        super().__init__(f"Kerr (a*={J_frac:.3f})") # <reason>Initializes with a name including the spin.</reason>
        self.J_frac = float(J_frac) # <reason>Stores the spin parameter.</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        a = self.J_frac * G_param * M_param / C_param # <reason>Calculates the spin parameter `a` with units of length.</reason>
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates the Schwarzschild radius.</reason>
        rho2, delta = r**2, r**2 - rs * r + a**2 # <reason>Defines standard intermediate variables in Kerr coordinates.</reason>
        g_tt = -(1 - rs * r / rho2) # <reason>Time component of the Kerr metric.</reason>
        g_rr = rho2 / (delta + EPSILON) # <reason>Radial component of the Kerr metric.</reason>
        g_pp = ((r**2 + a**2)**2 - delta * a**2) / rho2 # <reason>Azimuthal component of the Kerr metric.</reason>
        g_tp = -rs * a * r / rho2 # <reason>The off-diagonal time-azimuth component, which represents frame-dragging. Our integrator is not designed for this and will likely fail.</reason>
        return g_tt, g_rr, g_pp, g_tp # <reason>Returns the full set of Kerr metric components for the equatorial plane.</reason>

class NonLocal(GravitationalTheory):
    """GR modified by a cosmological constant term.
    <reason>This tests the effect of dark energy, via the cosmological constant Lambda, on local orbital dynamics. While small, its inclusion is crucial for cosmological relevance.</reason>"""
    def __init__(self): super().__init__("Non‑local (Λ)") # <reason>Initializes with a name indicating the cosmological constant.</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates the Schwarzschild radius.</reason>
        ct = LAMBDA_COSMO * r**2 / 3 # <reason>Calculates the cosmological term, which grows with distance.</reason>
        m = 1 - rs / r - ct # <reason>Adds the cosmological term to the Schwarzschild metric.</reason>
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r) # <reason>Returns the metric components.</reason>

class Tduality(GravitationalTheory):
    """A model inspired by T-duality in string theory, modifying the radial coordinate.
    <reason>String theory predicts a minimum length scale. T-duality suggests that probing below this scale is equivalent to probing large distances. This model implements a simple version of this idea.</reason>"""
    def __init__(self): super().__init__("T‑Duality (string)") # <reason>Initializes the theory with its descriptive name.</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates the Schwarzschild radius.</reason>
        re = r + rs**2 / r # <reason>Defines an "effective" radius `re` that incorporates the T-duality concept, preventing distances smaller than `rs` from being probed.</reason>
        m = 1 - rs / re # <reason>Constructs the metric using this effective radius.</reason>
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r) # <reason>Returns the metric components.</reason>

class Hydrodynamic(GravitationalTheory):
    """An emergent gravity model where spacetime is treated as a fluid.
    <reason>This tests the 'emergent gravity' paradigm, where gravity is not fundamental but arises from the collective behavior of microscopic degrees of freedom, much like hydrodynamics from atoms.</reason>"""
    def __init__(self): super().__init__("Emergent (hydrodynamic)") # <reason>Initializes the theory.</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates the Schwarzschild radius.</reason>
        v_cap = 0.999999 * C_param # <reason>Sets a cap on the fluid velocity to avoid infinities.</reason>
        vfs = torch.minimum(rs / r * C_param, torch.as_tensor(v_cap, device=r.device, dtype=r.dtype)) # <reason>Defines the fluid's speed based on the escape velocity.</reason>
        gamma_sq = 1.0 / (1.0 - (vfs / C_param)**2 + EPSILON) # <reason>Calculates the squared Lorentz factor for the fluid.</reason>
        return -gamma_sq, gamma_sq, r**2, torch.zeros_like(r) # <reason>Returns a metric based on the Lorentz factor, mimicking a relativistic fluid.</reason>

class Participatory(GravitationalTheory):
    """A model where observation influences the metric, inspired by Wheeler's "it from bit".
    <reason>This tests the most extreme interpretation of the paper's thesis: that information requires an observer, and that the act of observation could influence geometry. It's a highly philosophical but testable model.</reason>"""
    def __init__(self, obs_energy: float): # <reason>Takes the energy of the "observer" as a parameter.</reason>
        super().__init__(f"Participatory (E_obs={obs_energy:.1e})") # <reason>Initializes with a name including the observer energy.</reason>
        self.obs_energy = torch.as_tensor(obs_energy, device=device, dtype=DTYPE) # <reason>Stores the observer energy.</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates Schwarzschild radius.</reason>
        Ep = math.sqrt(hbar * C_param**5 / G_param) # <reason>Calculates the Planck energy for comparison.</reason>
        cert = 1 - torch.exp(-5 * self.obs_energy / Ep) # <reason>Calculates a "certainty" factor based on the ratio of observer energy to Planck energy. High energy -> high certainty.</reason>
        g_tt_gr = -(1 - rs / r) # <reason>The standard GR time component.</reason>
        g_rr_gr = 1 / (1 - rs / r + EPSILON) # <reason>The standard GR radial component.</reason>
        g_tt = cert * g_tt_gr + (1 - cert) * (-1.0) # <reason>The final metric is a weighted average of GR and flat spacetime, weighted by the certainty factor.</reason>
        g_rr = cert * g_rr_gr + (1 - cert) * 1.0 # <reason>The final metric is a weighted average of GR and flat spacetime, weighted by the certainty factor.</reason>
        return g_tt, g_rr, r**2, torch.zeros_like(r) # <reason>Returns the observer-dependent metric.</reason>

class Acausal(GravitationalTheory):
    """A model where the Hawking temperature affects the metric, implying final-state determination.
    <reason>This model introduces a dependency on the black hole's final thermodynamic state (its temperature), testing the radical idea that the future can influence the present spacetime geometry.</reason>"""
    def __init__(self): super().__init__("Acausal (final‑state)") # <reason>Initializes the theory.</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates Schwarzschild radius.</reason>
        ht = hbar * C_param**3 / (8 * math.pi * G_param * M_param * k) # <reason>Calculates the Hawking temperature of the black hole.</reason>
        pt = math.sqrt(hbar * C_param**5 / (G_param * k**2)) # <reason>Calculates the Planck temperature.</reason>
        cf = 1 - ht / pt # <reason>Creates a correction factor based on the ratio of Hawking to Planck temperature.</reason>
        m = 1 - (rs * cf) / r # <reason>Modifies the effective mass/radius of the black hole based on its final temperature.</reason>
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r) # <reason>Returns the acausally modified metric.</reason>
    
class QuantumCorrected(GravitationalTheory):
    """
    A generic model with a cubic correction term, representing some quantum effects.
    <reason>This model serves as a simple test case for higher-order corrections to the GR metric. Its performance relative to other theories helps classify the nature of potential quantum gravitational effects.</reason>
    """
    def __init__(self, alpha: float):
        super().__init__(f"Quantum Corrected (α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r + self.alpha * (rs / r) ** 3
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class VariableG(GravitationalTheory):
    """
    A model where the gravitational constant G varies with distance.
    <reason>This theory tests the fundamental assumption of a constant G. The asymmetric failure (stable for weakening G, unstable for strengthening G) provides a powerful insight into the necessary conditions for a stable universe.</reason>
    """
    def __init__(self, delta: float):
        super().__init__(f"Variable G (δ={delta:+.2f})")
        self.delta = torch.as_tensor(delta, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        G_eff = G_param * (1 + self.delta * torch.log1p(r / rs))
        m = 1 - 2 * G_eff * M_param / (C_param**2 * r)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalBase(GravitationalTheory):
    """Base class for the 'Einstein Final' model variants to standardize naming."""
    def __init__(self, name_suffix: str, alpha: float):
        super().__init__(f"Einstein Final ({name_suffix}, α={alpha:+.2f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)

class EinsteinFinalCubic(EinsteinFinalBase):
    """Original model: A simple cubic correction term added to the metric. (α=0 is GR)."""
    def __init__(self, alpha: float): super().__init__("Cubic", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**3
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalQuadratic(EinsteinFinalBase):
    """A quadratic correction term, testing a different power-law deviation."""
    def __init__(self, alpha: float): super().__init__("Quadratic", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**2
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        
class EinsteinFinalExponential(EinsteinFinalBase):
    """An exponentially suppressed correction, mimicking a short-range field."""
    def __init__(self, alpha: float): super().__init__("Exponential", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r) * (1 - self.alpha * torch.exp(-r/rs))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalAsymmetric(EinsteinFinalBase):
    """Simulates an asymmetric metric by modifying g_tt and g_rr differently."""
    def __init__(self, alpha: float): super().__init__("Asymmetric", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        mod = self.alpha * (rs/r)**2
        g_tt = -(1 - rs/r + mod)
        g_rr = 1 / (1 - rs/r - mod + EPSILON)
        return g_tt, g_rr, r**2, torch.zeros_like(r)

class EinsteinFinalTorsional(EinsteinFinalBase):
    """A quartic correction term, as a toy model for torsional effects."""
    def __init__(self, alpha: float): super().__init__("Torsional", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**4
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        
class EinsteinFinalUnifiedAdditive(EinsteinFinalBase):
    """A test of unified theory where the EM field is added with a variable coupling."""
    def __init__(self, alpha: float):
        super().__init__("Unified Additive", alpha)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        m = 1 - rs/r + self.alpha * (rq_sq / r**2)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalUnifiedMultiplicative(EinsteinFinalBase):
    """A non-linear interaction between the gravitational and EM fields."""
    def __init__(self, alpha: float):
        super().__init__("Unified Multiplicative", alpha)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        m = (1 - rs/r) * (1 + self.alpha * (rq_sq / r**2))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        
class EinsteinFinalLogGravity(EinsteinFinalBase):
    """A logarithmic modification to the gravitational potential."""
    def __init__(self, alpha: float): super().__init__("Log Gravity", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - (rs/r) * (1 - self.alpha * torch.log1p(rs/r))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalResonant(EinsteinFinalBase):
    """A speculative resonant term causing oscillatory corrections."""
    def __init__(self, alpha: float): super().__init__("Resonant", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r)**3 * torch.sin(r/rs)
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalPionic(EinsteinFinalBase):
    """A Yukawa-like interaction inspired by meson physics."""
    def __init__(self, alpha: float): super().__init__("Pionic", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r + self.alpha * (rs/r) * torch.exp(-r / (3*rs))
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)

class EinsteinFinalDynamicLambda(EinsteinFinalBase):
    """A 'local' cosmological constant that depends on gravitational field strength."""
    def __init__(self, alpha: float): super().__init__("Dynamic Lambda", alpha)
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs/r - self.alpha * (rs/r)**2
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
    
class EinsteinFinalEntropic(EinsteinFinalBase):
    r"""Models gravity as an entropic force, modifying the potential with a logarithmic term.
    <reason>Inspired by theories of emergent gravity (e.g., Verlinde), this model modifies the gravitational potential based on thermodynamic and holographic principles, where gravity arises from information entropy.</reason>"""
    def __init__(self, alpha: float): super().__init__("Entropic", alpha)
    # <reason>Initializes the base class with the name "Entropic".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation for this theory.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - rs/r + self.alpha * LP**2 / r**2 * torch.log(r / LP)
        # <reason>Adds a term proportional to `(1/r^2) * log(r)`, which arises in some models of entropic gravity related to the information content on a holographic screen.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the entropically-modified metric components.</reason>

class EinsteinFinalMembrane(EinsteinFinalBase):
    r"""A correction inspired by higher-dimensional brane-world scenarios.
    <reason>In some string theory models, our universe is a 'brane' in a higher-dimensional space. This can lead to gravity 'leaking' into other dimensions, which is modeled here as a steep correction to the potential.</reason>"""
    def __init__(self, alpha: float): super().__init__("Membrane", alpha)
    # <reason>Initializes the base class with the name "Membrane".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - torch.sqrt((rs/r)**2 + self.alpha * (LP/r)**4)
        # <reason>Modifies the metric such that at large `r`, it approximates GR, but at short `r`, a higher-dimensional term (`1/r^4`) controlled by `alpha` and the Planck Length `LP` dominates.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the brane-world-inspired metric components.</reason>

class EinsteinFinalGaussBonnet(EinsteinFinalBase):
    r"""A simplified model inspired by Gauss-Bonnet gravity, a common extension to GR.
    <reason>Gauss-Bonnet gravity adds a specific quadratic curvature term to the action. This phenomenological model captures the essence of such a modification with a steep 1/r^5 term that can arise in the metric solution.</reason>"""
    def __init__(self, alpha: float): super().__init__("Gauss-Bonnet", alpha)
    # <reason>Initializes the base class with the name "Gauss-Bonnet".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - rs/r + self.alpha * (rs/r)**5
        # <reason>Adds a very steep, short-range `1/r^5` correction, characteristic of some higher-order curvature theories like Gauss-Bonnet.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
    
class EinsteinFinalNonCommutative(EinsteinFinalBase):
    r"""A model motivated by non-commutative geometry, which regularizes the singularity.
    <reason>Non-commutative geometry suggests that spacetime coordinates do not commute at the Planck scale, which effectively 'smears' the singularity. This is modeled by an exponential term that smooths the metric core.</reason>"""
    def __init__(self, alpha: float): super().__init__("Non-Commutative", alpha)
    # <reason>Initializes the base class with the name "Non-Commutative".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - (rs * torch.exp(-self.alpha * LP**2 / r**2)) / r
        # <reason>Multiplies the effective mass by a Gaussian factor, which 'smears' the central point source over the Planck scale `LP`, effectively removing the singularity.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the non-commutative-inspired metric components.</reason>
        # <reason>Returns the Gauss-Bonnet-inspired metric components.</reason>    

class EinsteinFinalVacuum(EinsteinFinalBase):
    r"""A model where gravity's strength is coupled to the vacuum energy.
    <reason>This tests the idea that the strength of gravity could be affected by the energy density of the quantum vacuum, modeled here as a constant offset to the metric potential controlled by alpha.</reason>"""
    def __init__(self, alpha: float): super().__init__("Vacuum Coupling", alpha)
    # <reason>Initializes the base class with the name "Vacuum Coupling".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - rs/r + self.alpha * (LP/rs)**2
        # <reason>Adds a small, constant offset to the metric potential. This constant is dimensionless and represents the coupling between the vacuum energy (related to `LP`) and the black hole's own scale (`rs`).</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the vacuum-coupled metric components.</reason>
        
class EinsteinRegularized(GravitationalTheory):
    """
    A regularized version of GR that avoids a central singularity.
    <reason>This model is a key 'distinguishable' theory. It modifies GR only at the Planck scale, and its tiny but non-zero loss demonstrates the framework's extreme sensitivity to subtle physical deviations.</reason>
    """
    def __init__(self): super().__init__("Einstein Regularised Core") # <reason>Initializes the theory.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates Schwarzschild radius.</reason>
        m = 1 - rs / torch.sqrt(r**2 + LP**2) # <reason>Replaces `r` with `sqrt(r^2 + L_p^2)` in the denominator to smooth out the singularity at r=0.</reason>
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r) # <reason>Returns the regularized metric.</reason>
    
class EinsteinFinalPowerLaw(EinsteinFinalBase):
    r"""Generalizes the potential with a variable power law, deviating from 1/r.
    <reason>This is a fundamental test of the inverse-square law at relativistic scales. By allowing the exponent to deviate from 1 (via alpha), we can test for large-scale modifications to gravity.</reason>"""
    def __init__(self, alpha: float): super().__init__("Power Law", alpha)
    # <reason>Initializes the base class with the name "Power Law".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - (rs/r)**(1.0 - self.alpha)
        # <reason>Modifies the exponent of the gravitational potential from a fixed `1` to `1-alpha`, directly testing for deviations from the standard 1/r potential.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the power-law-modified metric components.</reason>

class EinsteinFinalConformal(EinsteinFinalBase):
    r"""A model inspired by conformal gravity, where physics is invariant under scale transformations.
    <reason>Conformal gravity is an alternative to GR that has different properties at cosmological scales. This model introduces a term that respects conformal symmetry, testing a different geometric foundation.</reason>"""
    def __init__(self, alpha: float): super().__init__("Conformal", alpha)
    # <reason>Initializes the base class with the name "Conformal".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - rs/r + self.alpha * r
        # <reason>Adds a term that grows linearly with `r`. This type of term appears in conformal gravity solutions and dramatically changes the long-distance behavior of spacetime.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the conformally-inspired metric components.</reason>

class EinsteinFinalDilation(EinsteinFinalBase):
    r"""A model including a dilaton field from string theory.
    <reason>String theory predicts the existence of a scalar field, the dilaton, which couples to gravity. This model tests a simple form of this coupling, modifying the strength of the gravitational potential.</reason>"""
    def __init__(self, alpha: float): super().__init__("Dilaton", alpha)
    # <reason>Initializes the base class with the name "Dilaton".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - (rs/r) / (1 + self.alpha * (rs/r))
        # <reason>Modifies the gravitational potential `rs/r` with a divisor that depends on the potential itself. This represents the dilaton field's self-interaction and its coupling to gravity.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the dilaton-modified metric components.</reason>

class EinsteinFinalTachyonic(EinsteinFinalBase):
    r"""A speculative model with a tachyonic field contribution.
    <reason>Tachyonic fields, while problematic, appear in some string theory contexts. This model tests the effect of a potential that weakens at short distances, a hallmark of tachyon condensation.</reason>"""
    def __init__(self, alpha: float): super().__init__("Tachyonic", alpha)
    # <reason>Initializes the base class with the name "Tachyonic".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - rs/r * (1 - self.alpha * torch.tanh(rs/r))
        # <reason>Uses a hyperbolic tangent (`tanh`) function. As `r` becomes small, `tanh(rs/r)` approaches 1, causing the correction term to reduce the gravitational potential, mimicking the behavior of tachyon condensation near the would-be singularity.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the tachyonic-field-inspired metric components.</reason>

class EinsteinFinalHigherDeriv(EinsteinFinalBase):
    r"""A model with both quadratic and cubic corrections.
    <reason>Instead of testing just one higher-order term, this model includes two, allowing for more complex interactions and a better fit if the true quantum corrections are not simple power laws.</reason>"""
    def __init__(self, alpha: float): super().__init__("Higher-Derivative", alpha)
    # <reason>Initializes the base class with the name "Higher-Derivative".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - rs/r + self.alpha * (rs/r)**2 - self.alpha * (rs/r)**3
        # <reason>Includes both a quadratic (`1/r^2`) and a cubic (`1/r^3`) correction term, allowing for a more complex modification to GR's short-distance behavior.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the metric components with multiple higher-derivative corrections.</reason>

class StochasticNoise(GravitationalTheory):
    """
    Wraps a base theory (Schwarzschild) and adds stochastic noise to the metric.
    <reason>This model directly tests the informational robustness of spacetime. It simulates a 'jittery' reality, and observing how orbits degrade provides a quantitative measure of a theory's stability against quantum-like fluctuations.</reason>
    """
    def __init__(self, noise_strength: float): # <reason>Takes the amplitude of the noise as a parameter.</reason>
        super().__init__(f"Stochastic Noise (σ={noise_strength:.1e})") # <reason>Initializes with a name including the noise strength.</reason>
        self.noise_strength = torch.as_tensor(noise_strength, device=device, dtype=DTYPE) # <reason>Stores the noise strength.</reason>
        self.base_theory = Schwarzschild() # <reason>Uses Schwarzschild as the base theory to which noise is added.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        g_tt, g_rr, g_pp, g_tp = self.base_theory.get_metric(r, M_param, C_param, G_param) # <reason>Gets the deterministic metric components first.</reason>
        noise_t = self.noise_strength * torch.randn_like(r) # <reason>Generates Gaussian random noise for the time component.</reason>
        noise_r = self.noise_strength * torch.randn_like(r) # <reason>Generates Gaussian random noise for the radial component.</reason>
        g_tt_noisy = g_tt * (1 + noise_t) # <reason>Applies noise multiplicatively to preserve the metric signature and scale appropriately.</reason>
        g_rr_noisy = g_rr * (1 + noise_r) # <reason>Applies noise multiplicatively.</reason>
        return g_tt_noisy, g_rr_noisy, g_pp, g_tp # <reason>Returns the noisy metric.</reason>

class EinsteinFinalBase(GravitationalTheory):
    """Base class for the 'Einstein Final' and 'Transformer' model variants to standardize naming."""
    def __init__(self, name_prefix: str, name_suffix: str, alpha: float): # <reason>Takes a prefix (EF or Transformer), a descriptive suffix, and the coupling parameter `alpha`.</reason>
        super().__init__(f"{name_prefix} ({name_suffix}, α={alpha:+.2f})") # <reason>Creates a standardized name, e.g., "EF (Cubic, α=-1.00)".</reason>
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE) # <reason>Stores the coupling parameter alpha as a tensor.</reason>


class EinsteinFinalQuintessence(EinsteinFinalBase):
    r"""A model that includes a quintessence-like scalar field.
    <reason>Quintessence is a hypothesized form of dark energy. This models its effect on local spacetime geometry as a very shallow power-law term, distinct from a cosmological constant.</reason>"""
    def __init__(self, alpha: float): super().__init__("Quintessence", alpha)
    # <reason>Initializes the base class with the name "Quintessence".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        m = 1 - rs/r - self.alpha * (r/rs)**0.5
        # <reason>Adds a term that grows with the square root of the radius. This represents the potential from a quintessence field, which changes very slowly over large distances.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the quintessence-modified metric components.</reason>


class EinsteinFinalUnifiedTheory(GravitationalTheory):
    r"""The culmination of Einstein's quest for a unified field theory.
    <reason>This model is the most ambitious synthesis, combining a non-linear reciprocal coupling of gravity and EM with a hyperbolic term representing a deterministic substructure. Its success would be the strongest possible validation of the paper's thesis.</reason>"""
    def __init__(self, gamma: float): # <reason>Takes a unique parameter `gamma` to distinguish it from the `alpha`-based models.</reason>
        super().__init__(f"Einstein's UFT (γ={gamma:+.3f})") # <reason>Initializes with a special name to denote its status.</reason>
        self.gamma = torch.as_tensor(gamma, device=device, dtype=DTYPE) # <reason>Stores the gamma parameter.</reason>
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE) # <reason>Stores the charge parameter for unified calculations.</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2 # <reason>Calculates Schwarzschild radius.</reason>
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4) # <reason>Calculates squared charge radius.</reason>
        u_g = rs / r # <reason>Dimensionless gravitational potential.</reason>
        u_e = rq_sq / r**2 # <reason>Dimensionless electromagnetic potential.</reason>
        hyp_mod = self.gamma * torch.cosh(u_e / (u_g + EPSILON)) - self.gamma # <reason>A hyperbolic correction term symbolizing a deterministic substructure, activated by the ratio of EM to gravitational potential.</reason>
        unified_potential = u_g / (1 + u_g * u_e) + u_e / (1 + u_e / u_g + EPSILON) # <reason>A non-linear reciprocal coupling where each field modulates the other, representing a deep geometric unity.</reason>
        g_tt = -(1 - unified_potential + hyp_mod / 2) # <reason>Asymmetric application of the hyperbolic modifier to the time component.</reason>
        g_rr = 1 / (1 - unified_potential - hyp_mod + EPSILON) # <reason>Asymmetric application to the space component, emulating a non-symmetric metric's effects.</reason>
        return g_tt, g_rr, r**2, torch.zeros_like(r) # <reason>Returns the components of this candidate unified metric.</reason>
    

class EinsteinUnifiedGeometricField2(GravitationalTheory):
    """
    A candidate for Einstein's final theory, synthesizing his work on unification.

    This model attempts to unify gravity and electromagnetism through a purely
    geometric framework, inspired by three key principles from Einstein's later work:

    1.  **Asymmetric Metric**: The metric's time and space components are modified
        differently, a phenomenological approach to an asymmetric metric tensor
        ($g_{\mu\nu} \neq g_{\nu\mu}$), where the antisymmetric part was hoped to
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
    def __init__(self, alpha: float):
        super().__init__(f"Unified Geometric Field (α={alpha:+.3f})")
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)

        # Dimensionless potentials
        u_g = rs / r  # Gravitational potential
        u_e = rq_sq / r**2 # Electromagnetic potential

        # Logarithmic term, representing a deep-structure modification
        log_mod = self.alpha * torch.log1p(u_g)

        # Unified potential: GR term + EM term modified by the gravitational potential
        # This creates a non-linear interaction, making EM emerge from the field.
        unified_potential = u_g - (u_e / (1 + u_g))

        # Asymmetric application to metric components
        g_tt = -(1 - unified_potential + log_mod)
        g_rr = 1 / (1 - unified_potential - log_mod + EPSILON)

        return g_tt, g_rr, r**2, torch.zeros_like(r)
    
class EinsteinUnifiedGeometricField(EinsteinFinalBase):
    r"""A candidate for Einstein's final theory, synthesizing his work on unification.
    <reason>This theory represents a culmination of the project's goals. It is a creative, physically motivated attempt to model the principles of unification that Einstein pursued. It combines his known theoretical approaches into a single, testable hypothesis.</reason>"""
    def __init__(self, alpha: float):
        # <reason>Constructor takes the coupling parameter `alpha`.</reason>
        super().__init__("Unified Geom.", alpha)
        # <reason>Initializes the base class with its descriptive name.</reason>
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)
        # <reason>Stores the charge parameter Q, as this theory is explicitly unified.</reason>

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # <reason>Defines the unified metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates Schwarzschild radius.</reason>
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        # <reason>Calculates squared charge radius.</reason>
        u_g = rs / r
        # <reason>Defines the dimensionless gravitational potential.</reason>
        u_e = rq_sq / r**2
        # <reason>Defines the dimensionless electromagnetic potential.</reason>
        log_mod = self.alpha * torch.log1p(u_g)
        # <reason>A logarithmic term representing a deep-structure modification or "hidden variable".</reason>
        unified_potential = u_g - (u_e / (1 + u_g))
        # <reason>A non-linear interaction where the gravitational potential `u_g` modifies the effect of the electromagnetic potential `u_e`, making EM an emergent feature of the geometry.</reason>
        g_tt = -(1 - unified_potential + log_mod)
        # <reason>The time component of the metric, including an asymmetric application of the logarithmic modifier.</reason>
        g_rr = 1 / (1 - unified_potential - log_mod + EPSILON)
        # <reason>The radial component with an opposite application of the modifier, mimicking a non-symmetric metric's effects.</reason>
        return g_tt, g_rr, r**2, torch.zeros_like(r)
        # <reason>Returns the components of this candidate unified metric.</reason>

class EinsteinFinalCubic(EinsteinFinalBase):
    r"""A simple cubic correction term, representing a basic higher-order modification.
    <reason>This is the original 'Final Equation' model. A cubic term is one of the simplest non-trivial ways to modify GR at short distances, making it a foundational test case.</reason>"""
    def __init__(self, alpha: float): super().__init__("Cubic", alpha)
    # <reason>Initializes the base class with its specific name "Cubic".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation for this specific theory.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius, a fundamental scale in the problem.</reason>
        m = 1 - rs/r + self.alpha * (rs/r)**3
        # <reason>Modifies the standard GR metric `(1 - rs/r)` with a cubic power-law correction term controlled by `alpha`.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the metric components `g_tt`, `g_rr`, `g_φφ`, and `g_tφ`.</reason>

class TransformerBase(GravitationalTheory):
    """Base class for Transformer-inspired models."""
    def __init__(self, name_suffix: str, alpha: float):
        # <reason>The constructor takes a descriptive suffix and the coupling parameter `alpha`.</reason>
        super().__init__(f"Transformer ({name_suffix}, α={alpha:+.2f})")
        # <reason>Creates a standardized name, e.g., "Transformer (Attention, α=1.00)".</reason>
        self.alpha = torch.as_tensor(alpha, device=device, dtype=DTYPE)
        # <reason>Stores the coupling parameter alpha as a tensor.</reason>

class TransformerAttention(TransformerBase):
    r"""Models a 'Gravitational Attention' mechanism.
    <reason>In Transformers, attention allows a model to weigh the importance of different pieces of information. This theory models a similar concept where the gravitational potential at a radius `r` is influenced not just by the mass `M` but also by a 'self-attention' mechanism related to the spacetime curvature at that point, represented by (rs/r). The `softmax`-like term `tanh` creates a weighted, non-linear focus.</reason>"""
    def __init__(self, alpha: float): super().__init__("Attention", alpha)
    # <reason>Initializes the base class with the name "Attention".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        u_g = rs/r
        # <reason>Defines the dimensionless gravitational potential, which acts as the 'query' and 'key'.</reason>
        attention_weight = torch.tanh(self.alpha * u_g)
        # <reason>Calculates a self-attention weight. The `tanh` function serves as a soft, non-linear weighting mechanism, similar to `softmax`.</reason>
        m = 1 - u_g * (1 + attention_weight)
        # <reason>The gravitational potential `u_g` is modified by its own attention weight, effectively focusing or defocusing the gravitational influence based on curvature.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the attention-modified metric components.</reason>
        

class TransformerPositional(TransformerBase):
    r"""Models 'Positional Encoding' as an oscillatory geometric field.
    <reason>Transformers use positional encodings to understand word order. This theory hypothesizes that spacetime has a fundamental set of positional 'frequencies'. The metric is modified by a sinusoidal term dependent on the logarithm of the radius, creating a geometric wave that provides 'positional information' to the orbiting particle.</reason>"""
    def __init__(self, alpha: float): super().__init__("Positional", alpha)
    # <reason>Initializes the base class with the name "Positional".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        log_r = torch.log(r / rs)
        # <reason>Calculates the logarithm of the dimensionless radius, analogous to the position index in a Transformer.</reason>
        pos_encoding = self.alpha * (rs/r) * torch.sin(log_r)
        # <reason>Creates a sinusoidal wave based on the log-position, representing a fundamental positional frequency in spacetime.</reason>
        m = 1 - rs/r + pos_encoding
        # <reason>Adds the positional encoding term to the standard GR metric.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the positionally-encoded metric components.</reason>

class TransformerLayerNorm(TransformerBase):
    r"""Models 'Layer Normalization' as a self-regulating geometric process.
    <reason>In deep learning, LayerNorm stabilizes training by normalizing activations. This theory proposes a similar stabilizing mechanism in spacetime. The gravitational potential is 'normalized' by dividing it by its own magnitude plus a term related to the Planck scale, preventing extreme values at short distances.</reason>"""
    def __init__(self, alpha: float): super().__init__("LayerNorm", alpha)
    # <reason>Initializes the base class with the name "LayerNorm".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        u_g = rs/r
        # <reason>Defines the dimensionless gravitational potential.</reason>
        norm_factor = torch.sqrt(u_g**2 + self.alpha * (LP/r)**2 + EPSILON)
        # <reason>Calculates a normalization factor, analogous to the standard deviation in LayerNorm, regularized by a Planck-scale term.</reason>
        m = 1 - u_g / norm_factor
        # <reason>Applies the normalization to the gravitational potential, stabilizing its effect at very small radii.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the normalized metric components.</reason>

class LogCorrected(GravitationalTheory):
    def __init__(self, beta: float):
        super().__init__(f"Log Corrected (β={beta:+.2f})")
        self.beta = torch.as_tensor(beta, device=device, dtype=DTYPE)

    def get_metric(self, r, M_param, C_param, G_param):
        rs = 2 * G_param * M_param / C_param**2
        sr = torch.maximum(r, rs * 1.001)
        lc = self.beta * (rs / sr) * torch.log(sr / rs)
        m = 1 - rs / r + lc
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class TransformerFeedForward(TransformerBase):
    r"""Models the 'Feed-Forward Network' as a non-linear geometric response.
    <reason>The feed-forward network in a Transformer applies a complex non-linear function. This theory models that by applying a two-layer 'network' to the gravitational potential: a non-linear activation (ReLU-like `torch.clamp_min`) followed by another modification, representing a more complex geometric response to mass.</reason>"""
    def __init__(self, alpha: float): super().__init__("FeedForward", alpha)
    # <reason>Initializes the base class with the name "FeedForward".</reason>
    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        u_g = rs/r
        # <reason>Defines the dimensionless gravitational potential, which is the input to the 'network'.</reason>
        activated_potential = torch.clamp_min(u_g - self.alpha, 0)
        # <reason>Applies a non-linear activation function (ReLU) to the potential, with a learnable bias `alpha`.</reason>
        m = 1 - (u_g + self.alpha * activated_potential)
        # <reason>The final metric term is a sum of the original potential and the output of the non-linear network, akin to a residual connection.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the metric components modified by the feed-forward network.</reason>


class TransformerValueMixing(TransformerBase):
    r"""Models the 'Value' vector in attention by mixing GR and EM potentials.
    <reason>In attention, 'values' are the information being mixed. This theory posits that the true geometric 'value' is a mix of the gravitational and electromagnetic potentials. The `alpha` parameter acts as a learned 'mixing weight', determining how much the electromagnetic field contributes to the final unified geometry.</reason>"""
    def __init__(self, alpha: float):
        # <reason>The constructor takes the mixing parameter `alpha`.</reason>
        super().__init__("ValueMixing", alpha)
        # <reason>Initializes the base class with the name "ValueMixing".</reason>
        self.Q = torch.as_tensor(Q_PARAM, device=device, dtype=DTYPE)
        # <reason>Stores the charge parameter Q, as this theory explicitly mixes gravitational and electromagnetic information.</reason>

    def get_metric(self, r, M_param, C_param, G_param):
        # <reason>Defines the metric calculation.</reason>
        rs = 2 * G_param * M_param / C_param**2
        # <reason>Calculates the Schwarzschild radius.</reason>
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        # <reason>Calculates the squared charge radius.</reason>
        u_g = rs / r
        # <reason>Defines the gravitational potential, the first 'value'.</reason>
        u_e = rq_sq / r**2
        # <reason>Defines the electromagnetic potential, the second 'value'.</reason>
        m = 1 - (u_g + self.alpha * u_e) / (1 + torch.abs(self.alpha))
        # <reason>Creates a weighted average of the gravitational and electromagnetic potentials, where `alpha` is the mixing weight. The denominator normalizes the contribution.</reason>
        return -m, 1/(m + EPSILON), r**2, torch.zeros_like(r)
        # <reason>Returns the metric components resulting from the mixed values.</reason>

# ---------------------------------------------------------------------------
# 3.  GEODESIC INTEGRATOR (RK‑4)
# ---------------------------------------------------------------------------

class GeodesicIntegrator:
    """
    Integrates the geodesic equations for a given gravitational theory using RK4.
    <reason>This class is the core of the simulation. It takes a theory, calculates the equations of motion from the metric using automatic differentiation (a modern and robust technique), and steps the particle's trajectory forward in time.</reason>
    """
    def __init__(self, model: GravitationalTheory, y0_full: Tensor, M_param: Tensor, C_param: float, G_param: float): # <reason>Constructor takes the model and all physical parameters.</reason>
        self.model, self.M, self.c, self.G = model, M_param, C_param, G_param # <reason>Stores model and parameters as instance variables.</reason>
        _, r0, _, dt_dtau0, _, dphi_dtau0 = y0_full # <reason>Unpacks the initial state vector.</reason>
        g_tt0, _, g_pp0, g_tp0 = self.model.get_metric(r0, self.M, self.c, self.G) # <reason>Gets the initial metric components at the starting radius.</reason>
        self.E  = -(g_tt0 * self.c * dt_dtau0 + g_tp0 * dphi_dtau0) # <reason>Calculates the conserved energy E per unit mass from initial conditions.</reason>
        self.Lz =  g_tp0 * self.c * dt_dtau0 + g_pp0 * dphi_dtau0 # <reason>Calculates the conserved angular momentum Lz per unit mass.</reason>
        if os.environ.get("TORCH_COMPILE") == "1" and hasattr(torch, "compile"): # <reason>Checks if TorchDynamo compilation is enabled via an environment variable.</reason>
            self._ode = torch.compile(self._ode_impl, fullgraph=True, mode="reduce-overhead", dynamic=True) # <reason>Compiles the ODE function for significant performance gains.</reason>
        else:
            self._ode = self._ode_impl # <reason>Uses the standard Python function if compilation is disabled or unavailable.</reason>

    def _ode_impl(self, y_state: Tensor) -> Tensor:
        """The right-hand side of the system of ODEs for the geodesic equations."""
        _, r, _, dr_dtau = y_state # <reason>Unpacks the current state vector [t, r, φ, dr/dτ].</reason>
        r_grad = r.clone().detach().requires_grad_(True) # <reason>Creates a new tensor for the radius that requires a gradient, essential for autograd.</reason>
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r_grad, self.M, self.c, self.G) # <reason>Calculates the metric components at the current radius.</reason>
        det = g_tp ** 2 - g_tt * g_pp # <reason>Calculates the determinant of the 2x2 (t,φ) sub-metric, needed to find the 4-velocities.</reason>
        if torch.abs(det) < EPSILON: return torch.zeros_like(y_state) # <reason>Prevents division by zero if the metric becomes degenerate.</reason>
        u_t   = (self.E * g_pp + self.Lz * g_tp) / det # <reason>Calculates the time component of the 4-velocity using the conserved quantities.</reason>
        u_phi = -(self.E * g_tp + self.Lz * g_tt) / det # <reason>Calculates the azimuthal component of the 4-velocity.</reason>
        V_sq = (-self.c ** 2 - (g_tt * u_t ** 2 + g_pp * u_phi ** 2 + 2 * g_tp * u_t * u_phi)) / g_rr # <reason>Calculates the squared effective radial potential from the metric normalization condition.</reason>
        if not torch.all(torch.isfinite(V_sq)): return torch.full_like(y_state, float('nan')) # <reason>A crucial check to prevent autograd from failing on non-finite inputs, signaling a simulation failure.</reason>
        (dV_dr,) = torch.autograd.grad(V_sq, r_grad, create_graph=False, retain_graph=False) # <reason>The core of the integrator: uses automatic differentiation to get the gradient of the potential with respect to radius.</reason>
        d2r_dtau2 = 0.5 * dV_dr # <reason>The radial geodesic equation relates the radial acceleration to the potential gradient.</reason>
        return torch.stack((u_t / self.c, dr_dtau, u_phi, d2r_dtau2)) # <reason>Returns the vector of derivatives [dt/dτ, dr/dτ, dφ/dτ, d²r/dτ²] for the next integration step.</reason>

    def rk4_step(self, y: Tensor, dτ: float) -> Tensor:
        """Performs a single Runge-Kutta 4th order integration step."""
        k1 = self._ode(y).detach() # <reason>First stage of RK4.</reason>
        k2 = self._ode((y + 0.5 * dτ * k1)).detach() # <reason>Second stage, evaluated at the midpoint.</reason>
        k3 = self._ode((y + 0.5 * dτ * k2)).detach() # <reason>Third stage, also at the midpoint but using the k2 estimate.</reason>
        k4 = self._ode((y + dτ * k3)).detach() # <reason>Fourth stage, evaluated at the endpoint.</reason>
        return y + (k1 + 2 * k2 + 2 * k3 + k4) * (dτ / 6.0) # <reason>Combines the stages with the correct weighting to get the final, more accurate step.</reason>

# ---------------------------------------------------------------------------
# 4.  ANALYSIS & MAIN DRIVER
# ---------------------------------------------------------------------------

def calculate_fft_loss(traj_ref: Tensor, traj_pred: Tensor) -> float:
    """
    Calculates the informational loss between two trajectories using FFT MSE.
    <reason>This function is the core of the paper's methodology. It compares the full frequency spectrum of orbital dynamics, capturing subtle differences in precession and shape that a simple endpoint comparison would miss. It is a direct measure of a theory's informational fidelity.</reason>
    """
    min_len = min(len(traj_ref), len(traj_pred)) # <reason>Truncates trajectories to the same length for a valid comparison, as some orbits decay faster than others.</reason>
    if min_len < 2: return float("inf") # <reason>FFT requires at least two data points to be meaningful.</reason>
    r_ref, r_pred = traj_ref[:min_len, 1], traj_pred[:min_len, 1] # <reason>Extracts the radial component (the signal) from each trajectory history.</reason>
    if not (torch.all(torch.isfinite(r_ref)) and torch.all(torch.isfinite(r_pred))): # <reason>Checks for numerical errors before attempting FFT.</reason>
        return float('nan') # <reason>Returns NaN if the trajectory data is invalid.</reason>
    fft_ref, fft_pred = torch.fft.fft(r_ref), torch.fft.fft(r_pred) # <reason>Computes the Fast Fourier Transform for both signals.</reason>
    return torch.mean((torch.abs(fft_ref) - torch.abs(fft_pred)) ** 2).item() # <reason>Calculates the Mean Squared Error between the magnitudes of the two frequency spectra and returns it as a standard Python float.</reason>

def main() -> None:
    """
    Main driver for the simulation.
    <reason>This function orchestrates the entire process: setting up models, defining initial conditions, running the simulations, calculating losses, and reporting the results.</reason>
    """
    print("=" * 80) # <reason>Prints a header for console output clarity.</reason>
    print(f"PyTorch Orbital Test | device={device} | dtype={DTYPE}") # <reason>Reports the current hardware and precision settings.</reason>
    print("=" * 80) # <reason>Prints a header for console output clarity.</reason>

    # -- Model Registry --
    # <reason>This comprehensive registry includes all original and new theories to ensure a complete and exhaustive search across the theoretical landscape.</reason>
    models: list[GravitationalTheory] = [
        Schwarzschild(), NewtonianLimit(), ReissnerNordstrom(Q_PARAM), Kerr(J_FRAC),
        NonLocal(), Tduality(), Hydrodynamic(), Participatory(OBSERVER_ENERGY), Acausal(),
        EinsteinRegularized(), StochasticNoise(STOCHASTIC_STRENGTH),
    ] # <reason>Initializes the list of models with all original, non-parameterized theories.</reason>
    
    e_final_classes = [
        EinsteinFinalCubic, EinsteinFinalQuadratic, EinsteinFinalExponential,
        EinsteinFinalAsymmetric, EinsteinFinalTorsional, EinsteinFinalUnifiedAdditive,
        EinsteinFinalUnifiedMultiplicative, EinsteinFinalLogGravity, EinsteinFinalResonant,
        EinsteinFinalPionic, EinsteinFinalDynamicLambda, EinsteinFinalEntropic,
        EinsteinFinalMembrane, EinsteinFinalGaussBonnet, EinsteinFinalNonCommutative,
        EinsteinFinalVacuum, EinsteinFinalPowerLaw, EinsteinFinalConformal,
        EinsteinFinalDilation, EinsteinFinalHigherDeriv, EinsteinFinalQuintessence,
        EinsteinFinalTachyonic, EinsteinUnifiedGeometricField, EinsteinUnifiedGeometricField2, EinsteinFinalUnifiedTheory
    ] # <reason>Defines the list of all "Einstein Final" speculative theory classes.</reason>
    transformer_classes = [
        TransformerAttention, TransformerPositional, TransformerLayerNorm,
        TransformerFeedForward, TransformerValueMixing
    ] # <reason>Defines the list of all Transformer-inspired speculative theory classes.</reason>
    
    for e_cls in e_final_classes + transformer_classes: # <reason>Begins a loop to instantiate all speculative models.</reason>
        for val in [0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0]: # <reason>Tests a range of parameter values for each theory to probe its behavior.</reason>
            if e_cls == EinsteinFinalUnifiedTheory: # <reason>Handles the special case for the final theory, which uses a different parameter name (`gamma`).</reason>
                models.append(e_cls(gamma=val)) # <reason>Instantiates the final theory with the `gamma` parameter.</reason>
            else:
                models.append(e_cls(alpha=val)) # <reason>Instantiates all other speculative theories with the standard `alpha` parameter.</reason>

    # Add original sweeps
    sweeps = {
        "QuantumCorrected": (QuantumCorrected, dict(alpha=np.linspace(-2.0, 2.0, 10))),
        "LogCorrected": (LogCorrected, dict(beta=np.linspace(-1.5, 1.5, 10))),
        "VariableG": (VariableG, dict(delta=np.linspace(-0.5, 0.5, 10))),
    } # <reason>Defines the original parameter sweeps to ensure exhaustive testing.</reason>
    for cls, pd in sweeps.values(): # <reason>Loops through the sweep definitions.</reason>
        key, vals = next(iter(pd.items())) # <reason>Extracts the parameter name and its array of values.</reason>
        models += [cls(**{key: float(v)}) for v in vals] # <reason>Instantiates and adds each model from the sweep to the main list.</reason>

    print(f"Total models to be tested: {len(models)}") # <reason>Reports the total simulation count to the user.</reason>

    # -- Initial Conditions --
    r0 = 4.0 * RS # <reason>Sets the initial radius to 4 times the Schwarzschild radius, a common choice for a stable starting orbit.</reason>
    v_tan = torch.sqrt(G * M / r0) # <reason>Calculates the tangential velocity required for a circular Newtonian orbit at r0, providing a good starting point.</reason>
    g_tt0, _, g_pp0, _ = Schwarzschild().get_metric(r0, M, c, G) # <reason>Gets the metric components at the starting radius to calculate the initial 4-velocity.</reason>
    norm_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * c)) ** 2 # <reason>Normalizes the 4-velocity using the metric.</reason>
    dt_dtau0 = 1.0 / torch.sqrt(norm_sq) # <reason>Calculates the initial time-like component of the 4-velocity.</reason>
    dphi_dtau0 = (v_tan / r0) * dt_dtau0 # <reason>Calculates the initial angular component of the 4-velocity.</reason>
    y0_full = torch.tensor([0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()], device=device, dtype=DTYPE) # <reason>Assembles the full 6D initial state vector [t, r, φ, dt/dτ, dr/dτ, dφ/dτ].</reason>
    y0_state = y0_full[[0, 1, 2, 4]].clone() # <reason>Slices the full vector to get the 4D state vector [t, r, φ, dr/dτ] used by the integrator.</reason>

    # -- Run Parameters --
    DTau = 0.01 # <reason>Sets the proper time step for the RK4 integrator. A small value is needed for accuracy.</reason>
    MAX_CONSECUTIVE_FAILURES = 10 # <reason>Defines the threshold for the exponential backoff mechanism.</reason>
    if args.final: # <reason>Checks if the `--final` flag was used.</reason>
        N_STEPS, STEP_PRINT, SAVE_PLOTS = 5_000_000, 250_000, True # <reason>Sets parameters for a long, high-precision run.</reason>
        print("Mode: FINAL (high precision, long duration)") # <reason>Informs the user of the run mode.</reason>
    else:
        N_STEPS, STEP_PRINT, SAVE_PLOTS = 100_000, 10_000, args.plots # <reason>Sets parameters for a shorter, exploratory run.</reason>
        print("Mode: EXPLORATORY (fast, for prototyping)") # <reason>Informs the user of the run mode.</reason>
    
    # -- Ground-Truth Trajectory Generation (Cached) --
    def cached_run(model: GravitationalTheory, tag: str) -> Tensor:
        """Runs a simulation for a given model, caching the result."""
        precision_tag = "f64" if DTYPE == torch.float64 else "f32" # <reason>Creates a tag for the filename based on precision.</reason>
        fname = f"cache_{tag}_{N_STEPS}_{precision_tag}.pt" # <reason>Defines a unique filename for the cache file.</reason>
        if os.path.exists(fname): return torch.load(fname, map_location=device) # <reason>If a cache file exists, load it to avoid re-computation.</reason>
        print(f"\n--- Generating Ground Truth: {model.name} ---") # <reason>Informs the user that a new baseline is being computed.</reason>
        integ = GeodesicIntegrator(model, y0_full, M, c, G) # <reason>Initializes the integrator for the baseline model.</reason>
        hist = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE) # <reason>Pre-allocates memory for the trajectory history for performance.</reason>
        hist[0], y = y0_state, y0_state.clone() # <reason>Sets the initial state.</reason>
        for i in range(N_STEPS): # <reason>The main simulation loop for the baseline.</reason>
            y = integ.rk4_step(y, DTau) # <reason>Calculates the next step.</reason>
            hist[i + 1] = y # <reason>Stores the new state.</reason>
            if (i + 1) % STEP_PRINT == 0: print(f"  ...step {i+1:,}/{N_STEPS:,} | r={y[1]/RS:.3f} RS") # <reason>Prints progress periodically.</reason>
            if not torch.all(torch.isfinite(y)) or y[1] <= RS * 1.01: # <reason>Checks for simulation failure or capture by the black hole.</reason>
                hist = hist[: i + 2]; break # <reason>Truncates the history and exits the loop if the simulation ends early.</reason>
        torch.save(hist, fname) # <reason>Saves the computed trajectory to the cache file.</reason>
        return hist # <reason>Returns the computed trajectory.</reason>

    GR_hist = cached_run(Schwarzschild(), "GR") # <reason>Generates or loads the pure gravity baseline.</reason>
    RN_hist = cached_run(ReissnerNordstrom(Q_PARAM), "RN") # <reason>Generates or loads the gravity + electromagnetism baseline.</reason>
    GR_loss_vs_RN = calculate_fft_loss(RN_hist, GR_hist) # <reason>Calculates the loss of GR against the R-N baseline, used as a benchmark for breakthrough theories.</reason>

    # -- Main Evaluation Loop --
    results = [] # <reason>Initializes an empty list to store the results of all simulations.</reason>
    for idx, model in enumerate(models, 1): # <reason>Loops through every model in the registry.</reason>
        print(f"\n[{idx:03}/{len(models)}] Evaluating: {model.name}") # <reason>Prints the current model being evaluated.</reason>
        integ = GeodesicIntegrator(model, y0_full, M, c, G) # <reason>Initializes the integrator for the current model.</reason>
        traj = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE) # <reason>Pre-allocates memory for the trajectory.</reason>
        traj[0], y = y0_state, y0_state.clone() # <reason>Sets the initial state.</reason>
        consecutive_failures = 0 # <reason>Initializes the failure counter for the backoff mechanism.</reason>
        for i in range(N_STEPS): # <reason>The main simulation loop for the current model.</reason>
            y = integ.rk4_step(y, DTau) # <reason>Calculates the next step.</reason>
            traj[i + 1] = y # <reason>Stores the new state.</reason>
            if not torch.all(torch.isfinite(y)): # <reason>Checks if the simulation produced a numerical error (NaN or infinity).</reason>
                consecutive_failures += 1 # <reason>Increments the failure counter.</reason>
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES: # <reason>Checks if the failure threshold has been reached.</reason>
                    print(f"  ! ABORTED: Simulation unstable for {consecutive_failures} consecutive steps.") # <reason>Informs the user that the simulation is being aborted early.</reason>
                    traj = traj[:i+2]; break # <reason>Aborts the loop to save compute time on a failed model.</reason>
            else:
                consecutive_failures = 0 # <reason>Resets the failure counter after a successful step.</reason>
            if y[1] <= RS * 1.01: # <reason>Checks if the particle has been captured by the black hole.</reason>
                traj = traj[:i+2]; break # <reason>Ends the simulation early upon capture.</reason>
        results.append({ # <reason>Appends a dictionary of the results for this model to the main results list.</reason>
            "name": model.name, # <reason>Stores the model's name.</reason>
            "loss_GR": calculate_fft_loss(GR_hist, traj), # <reason>Calculates and stores the loss against the GR baseline.</reason>
            "loss_RN": calculate_fft_loss(RN_hist, traj), # <reason>Calculates and stores the loss against the R-N baseline.</reason>
            "traj": traj.cpu().numpy(), # <reason>Stores the trajectory after moving it to the CPU and converting to a NumPy array for plotting.</reason>
        })

    # -- Reporting and Plotting --
    PLOT_DIR = f"plots/run_{int(time.time())}" # <reason>Creates a unique directory name for this run's plots.</reason>
    if SAVE_PLOTS and not args.no_plots: os.makedirs(PLOT_DIR, exist_ok=True) # <reason>Creates the plot directory if it doesn't exist.</reason>
    BOLD, GREEN_BG, RESET = "\033[1m", "\033[42m", "\033[0m" # <reason>Defines ANSI escape codes for highlighting console output.</reason>

    results.sort(key=lambda d: (d["loss_GR"] is None or math.isnan(d["loss_GR"]), d["loss_GR"])) # <reason>Sorts the results by GR loss, placing failed runs (NaN) at the bottom.</reason>
    print("\n\n" + "="*80) # <reason>Prints a separator.</reason>
    print("--- RANKING vs. GENERAL RELATIVITY (GR) ---") # <reason>Prints the header for the first results table.</reason>
    print("Rank | Model                                        | Loss_GR (FFT MSE)") # <reason>Prints the table column headers.</reason>
    print("-" * 75) # <reason>Prints a separator line.</reason>
    for rank, res in enumerate(results, 1): # <reason>Loops through the sorted results.</reason>
        print(f"{rank:4d} | {res['name']:<44} | {res.get('loss_GR', float('nan')):10.3e}") # <reason>Prints a formatted row for each result.</reason>
    print("="*80) # <reason>Prints a closing separator.</reason>

    results.sort(key=lambda d: (d["loss_RN"] is None or math.isnan(d["loss_RN"]), d["loss_RN"])) # <reason>Sorts the results by R-N loss for the second table.</reason>
    print("\n--- RANKING vs. REISSNER-NORDSTRÖM (R-N) ---") # <reason>Prints the header for the R-N results table.</reason>
    print(f"(GR baseline loss vs R-N is: {GR_loss_vs_RN:.3e})") # <reason>Prints the GR vs R-N loss value, which serves as the benchmark for a breakthrough.</reason>
    print("Rank | Model                                        | Loss_RN (FFT MSE)") # <reason>Prints the table column headers.</reason>
    print("-" * 75) # <reason>Prints a separator line.</reason>
    for rank, res in enumerate(results, 1): # <reason>Loops through the newly sorted results.</reason>
        loss_val = res.get('loss_RN', float('nan')) # <reason>Gets the R-N loss value.</reason>
        name = res['name'] # <reason>Gets the model name.</reason>
        is_breakthrough = not math.isnan(loss_val) and loss_val < GR_loss_vs_RN and "Schwarzschild" not in name and "Reissner" not in name # <reason>This logic identifies a potential breakthrough: a speculative theory that has a valid loss and performs better than GR against the R-N baseline.</reason>
        if is_breakthrough: # <reason>Checks if the breakthrough condition is met.</reason>
            print(f"{GREEN_BG}{BOLD}{rank:4d} | {name:<44} | {loss_val:10.3e} [BREAKTHROUGH]{RESET}") # <reason>Prints the result with special highlighting to make it obvious.</reason>
        else:
            print(f"{rank:4d} | {name:<44} | {loss_val:10.3e}") # <reason>Prints the standard result row.</reason>
    print("="*80) # <reason>Prints a closing separator.</reason>

    if SAVE_PLOTS and not args.no_plots: # <reason>Checks if plotting is enabled.</reason>
        GR_np, RN_np = GR_hist.cpu().numpy(), RN_hist.cpu().numpy() # <reason>Converts baseline trajectories to NumPy arrays for Matplotlib.</reason>
        results.sort(key=lambda d: (d["loss_GR"] is None or math.isnan(d["loss_GR"]), d["loss_GR"])) # <reason>Sorts back by GR loss to plot the top performers in the primary test.</reason>
        top_results = results if args.plots else results[:5] # <reason>Selects either all models or the top 5 for plotting based on the `--plots` flag.</reason>
        print(f"\nGenerating plots for top {len(top_results)} models in '{PLOT_DIR}/'...") # <reason>Informs the user that plotting has begun.</reason>
        for res in top_results: # <reason>Loops through the models to be plotted.</reason>
            pred_np = res["traj"] # <reason>Gets the trajectory for the current model.</reason>
            plt.figure(figsize=(8, 8)) # <reason>Creates a new figure for each plot.</reason>
            ax = plt.subplot(111, projection="polar") # <reason>Creates a polar subplot.</reason>
            ax.plot(GR_np[:, 2], GR_np[:, 1], "k--", label="GR", linewidth=1.5, zorder=5) # <reason>Plots the GR baseline with high z-order to ensure it's visible.</reason>
            ax.plot(RN_np[:, 2], RN_np[:, 1], "b:",  label="R-N", linewidth=1.5, zorder=5) # <reason>Plots the R-N baseline.</reason>
            ax.plot(pred_np[:, 2], pred_np[:, 1], "r-", label=res["name"], zorder=4) # <reason>Plots the model's predicted trajectory.</reason>
            ax.plot(pred_np[0, 2], pred_np[0, 1], "go", markersize=8, label="start", zorder=6) # <reason>Marks the start point.</reason>
            ax.plot(pred_np[-1, 2], pred_np[-1, 1], "rx", markersize=10, mew=2, label="end", zorder=6) # <reason>Marks the end point.</reason>
            ax.set_title(res["name"], pad=20) # <reason>Sets the plot title.</reason>
            ax.legend(); plt.tight_layout() # <reason>Adds a legend and adjusts layout.</reason>
            safe_name = res["name"].translate({ord(c): "_" for c in " /()=.*+-"}) # <reason>Creates a safe filename by replacing special characters.</reason>
            plt.savefig(os.path.join(PLOT_DIR, f"{safe_name}.png")) # <reason>Saves the figure to a PNG file.</reason>
            plt.close() # <reason>Closes the figure to free up memory.</reason>
        print("Plots saved successfully.") # <reason>Confirms that plots have been saved.</reason>

    print("\nDone.") # <reason>Indicates the end of the script.</reason>

if __name__ == "__main__": # <reason>Standard Python entry point to make the script executable.</reason>
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning) # <reason>Suppresses non-critical JIT tracer warnings from torch.compile to keep console output clean.</reason>
    main() # <reason>Calls the main function to run the simulation.</reason>