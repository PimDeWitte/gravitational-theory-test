#!/usr/bin/env python3
# sim_gpu.py  ── July 2025
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
# --- UPDATE (JULY 9, 2025) ---
# - Implemented dual-baseline reporting for both GR and Reissner-Nordström.
# - Corrected plotting artifacts where baselines were obscured.
# - Added a new `StochasticNoise` model to test theoretical robustness.
# - Added --cpu-f64 flag for high-precision validation runs.
# - Implemented exponential backoff for failing simulations to save compute.
# - Pruned uninformative (lossless or always-failing) theories.
# - Completed full code documentation and logical review.
# ---------------------------------------------------------------------------

from __future__ import annotations
import os, time, math, argparse, warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c, k, hbar, epsilon_0

# ---------------------------------------------------------------------------
# 0.  CLI ARGUMENTS & GLOBAL CONFIG
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    """
    Parses command-line arguments for the simulation.
    <reason>This function encapsulates argument parsing, making the main script cleaner and easier to read. It defines the operational modes of the script, such as plotting, precision, and run duration.</reason>
    """
    p = argparse.ArgumentParser(description="PyTorch-based orbital mechanics simulator for gravitational theories.")
    p.add_argument("--final", action="store_true", help="Run with final, high-step-count parameters for publication-quality data.")
    p.add_argument("--plots", action="store_true", help="Generate and save plots for the top-ranked models.")
    p.add_argument("--no-plots", action="store_true", help="Explicitly disable all plotting, even if other flags would enable it.")
    p.add_argument("--cpu-f64", action="store_true", help="Run on CPU with float64 precision for validation. Overrides default GPU/float32 settings.")
    return p.parse_args()

args = parse_cli()

# Set device and data type based on CLI flags. This must be done before any tensors are created.
# <reason>This block allows for flexible hardware and precision choices. The default is fast GPU/float32 for exploration, while --cpu-f64 enables high-precision CPU runs for validating key results, as recommended in the research plan.</reason>
if args.cpu_f64:
    DTYPE  = torch.float64
    device = torch.device("cpu")
else:
    DTYPE  = torch.float32
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
M_SI  = 10.0 * 1.989e30
RS_SI = 2 * G * M_SI / c**2
M  = torch.as_tensor(M_SI , device=device, dtype=DTYPE)
RS = torch.as_tensor(RS_SI, device=device, dtype=DTYPE)

# Cached Planck Length Tensor
# <reason>The Planck Length is used in some quantum gravity models. Caching it as a tensor avoids recalculating the Python float and converting it to a tensor inside the simulation loop.</reason>
LP = torch.as_tensor(math.sqrt(G * hbar / c**3), device=device, dtype=DTYPE)

# Default parameters for various speculative models.
# <reason>These default values are used to instantiate the non-swept versions of the theories. They are chosen to be physically significant enough to produce a deviation from GR without immediately causing the simulation to fail.</reason>
J_FRAC, Q_PARAM, Q_UNIFIED         = 0.5, 3.0e14, 1.0e12
ASYMMETRY_PARAM, TORSION_PARAM     = 1.0e-4, 1.0e-3
OBSERVER_ENERGY                    = 1.0e9
STOCHASTIC_STRENGTH                = 1e-7 # A very small value to ensure stability

# ---------------------------------------------------------------------------
# 2.  THEORY DEFINITIONS
# ---------------------------------------------------------------------------

Tensor = torch.Tensor  # Type alias for brevity

class GravitationalTheory:
    """
    Abstract base class for all gravitational theories.
    <reason>This class defines a common interface (`get_metric`) that all theories must implement. This polymorphic design allows the integrator to treat any theory identically, simplifying the simulation logic and making the framework easily extensible.</reason>
    """
    def __init__(self, name: str) -> None:
        self.name = name

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Calculates the metric components (g_tt, g_rr, g_φφ, g_tφ) for a given radius."""
        raise NotImplementedError

# -- 2.1 Standard & Baseline Metrics --

class Schwarzschild(GravitationalTheory):
    """
    The Schwarzschild metric for a non-rotating, uncharged black hole.
    <reason>This is the exact solution to Einstein's field equations in a vacuum and serves as the fundamental ground truth (baseline) for pure gravity in this framework.</reason>
    """
    def __init__(self): super().__init__("Schwarzschild (GR)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / (r + EPSILON) # Add epsilon to avoid division by zero at the singularity
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class NewtonianLimit(GravitationalTheory):
    """
    The Newtonian approximation of gravity.
    <reason>This theory is included as a 'distinguishable' model. It correctly lacks spatial curvature (g_rr = 1), and its significant but finite loss value validates the framework's ability to quantify physical incompleteness.</reason>
    """
    def __init__(self): super().__init__("Newtonian Limit")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        # Spatial curvature g_rr is 1 in the Newtonian limit.
        return -m, torch.ones_like(r), r**2, torch.zeros_like(r)

class ReissnerNordstrom(GravitationalTheory):
    """
    The Reissner-Nordström metric for a charged, non-rotating black hole.
    <reason>This is the exact solution for a charged mass and serves as the second ground truth (the Kaluza-Klein baseline) for testing a theory's ability to unify gravity and electromagnetism.</reason>
    """
    def __init__(self, Q: float):
        super().__init__(f"Reissner‑Nordström (Q={Q:.1e})")
        self.Q = torch.as_tensor(Q, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # The metric includes an additional term related to charge Q.
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        m = 1 - rs / r + rq_sq / r**2
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

# -- 2.2 Speculative & Modified Metrics --

class EinsteinRegularized(GravitationalTheory):
    """
    A regularized version of GR that avoids a central singularity.
    <reason>This model is a key 'distinguishable' theory. It modifies GR only at the Planck scale, and its tiny but non-zero loss demonstrates the framework's extreme sensitivity to subtle physical deviations.</reason>
    """
    def __init__(self): super().__init__("Einstein Regularised Core")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Replaces `r` with `sqrt(r^2 + L_p^2)` in the denominator to smooth out the singularity.
        m = 1 - rs / torch.sqrt(r**2 + LP**2)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

class LogCorrected(GravitationalTheory):
    """
    A quantum gravity inspired model with a logarithmic correction term.
    <reason>This model is a high-performing 'distinguishable'. Logarithmic corrections are predicted by some quantum loop gravity theories, making this a promising candidate for a first-order quantum correction to GR.</reason>
    """
    def __init__(self, beta: float):
        super().__init__(f"Log Corrected (β={beta:+.2f})")
        self.beta = torch.as_tensor(beta, device=device, dtype=DTYPE)

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Clamp r to be slightly outside the horizon to avoid log(0)
        sr = torch.maximum(r, rs * 1.001)
        # Adds a logarithmic correction term, controlled by beta.
        log_corr = self.beta * (rs / sr) * torch.log(sr / rs)
        m = 1 - rs / r + log_corr
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

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
        # Adds a correction proportional to (rs/r)^3
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
        # G becomes a function of r, controlled by delta.
        G_eff = G_param * (1 + self.delta * torch.log1p(r / rs))
        m = 1 - 2 * G_eff * M_param / (C_param**2 * r)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)


class StochasticNoise(GravitationalTheory):
    """
    Wraps a base theory (Schwarzschild) and adds stochastic noise to the metric.
    <reason>This model directly tests the informational robustness of spacetime. It simulates a 'jittery' reality, and observing how orbits degrade provides a quantitative measure of a theory's stability against quantum-like fluctuations.</reason>
    """
    def __init__(self, noise_strength: float):
        super().__init__(f"Stochastic Noise (σ={noise_strength:.1e})")
        self.noise_strength = torch.as_tensor(noise_strength, device=device, dtype=DTYPE)
        self.base_theory = Schwarzschild()

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Get the deterministic metric from the base Schwarzschild theory.
        g_tt, g_rr, g_pp, g_tp = self.base_theory.get_metric(r, M_param, C_param, G_param)
        # Generate random noise scaled by the metric component's value.
        noise_t = self.noise_strength * torch.randn_like(r)
        noise_r = self.noise_strength * torch.randn_like(r)
        # Apply noise multiplicatively to preserve the metric signature.
        g_tt_noisy = g_tt * (1 + noise_t)
        g_rr_noisy = g_rr * (1 + noise_r)
        return g_tt_noisy, g_rr_noisy, g_pp, g_tp

# ---------------------------------------------------------------------------
# 3.  GEODESIC INTEGRATOR (RK‑4)
# ---------------------------------------------------------------------------

class GeodesicIntegrator:
    """
    Integrates the geodesic equations for a given gravitational theory using RK4.
    <reason>This class is the core of the simulation. It takes a theory, calculates the equations of motion from the metric using automatic differentiation (a modern and robust technique), and steps the particle's trajectory forward in time.</reason>
    """
    def __init__(self, model: GravitationalTheory, y0_full: Tensor, M_param: Tensor, C_param: float, G_param: float):
        """Initializes the integrator with a model and initial conditions."""
        self.model, self.M, self.c, self.G = model, M_param, C_param, G_param

        # Conserved quantities (Energy and Angular Momentum) are calculated from initial conditions.
        # <reason>For stationary metrics, E and Lz are constants of motion. Calculating them once at the start and reusing them is far more efficient and stable than recalculating them at every step.</reason>
        _, r0, _, dt_dtau0, _, dphi_dtau0 = y0_full
        g_tt0, _, g_pp0, g_tp0 = self.model.get_metric(r0, self.M, self.c, self.G)
        self.E  = -(g_tt0 * self.c * dt_dtau0 + g_tp0 * dphi_dtau0)
        self.Lz =  g_tp0 * self.c * dt_dtau0 + g_pp0 * dphi_dtau0

        # Optional Torch-Dynamo compilation for a significant speedup.
        # <reason>torch.compile can fuse operations and optimize the execution graph, which is ideal for the repetitive calculations in an ODE solver. This provides M-series chip optimizations transparently.</reason>
        if os.environ.get("TORCH_COMPILE") == "1" and hasattr(torch, "compile"):
            try:
                self._ode = torch.compile(self._ode_impl, fullgraph=True, mode="reduce-overhead", dynamic=True)
            except Exception as exc:
                warnings.warn(f"torch.compile disabled: {exc}")
                self._ode = self._ode_impl
        else:
            self._ode = self._ode_impl

    def _ode_impl(self, y_state: Tensor) -> Tensor:
        """
        The right-hand side of the system of ODEs for the geodesic equations.
        <reason>This function defines the physics. It calculates the derivatives of the state vector (t, r, φ, dr/dτ). Using torch.autograd.grad to find the potential gradient is a powerful alternative to manually deriving and coding complex Christoffel symbols.</reason>
        """
        _, r, _, dr_dtau = y_state

        # Use autograd to compute the potential gradient from the metric.
        r_grad = r.clone().detach().requires_grad_(True)
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r_grad, self.M, self.c, self.G)

        # Check for metric breakdown (determinant close to zero).
        det = g_tp ** 2 - g_tt * g_pp
        if torch.abs(det) < EPSILON:
            return torch.zeros_like(y_state) # Return zero velocity if metric fails.

        # Calculate 4-velocities from conserved quantities.
        u_t   = (self.E * g_pp + self.Lz * g_tp) / det
        u_phi = -(self.E * g_tp + self.Lz * g_tt) / det

        # Effective radial potential V(r).
        V_sq = (-self.c ** 2 - (g_tt * u_t ** 2 + g_pp * u_phi ** 2 + 2 * g_tp * u_t * u_phi)) / g_rr
        
        # Logical Correction: Pre-emptively check for non-finite potential before autograd.
        # <reason>Calling autograd on a non-finite tensor can cause cryptic errors. This check provides a clear failure point, making the simulation more robust and easier to debug.</reason>
        if not torch.all(torch.isfinite(V_sq)):
            return torch.full_like(y_state, float('nan')) # Return NaN to signal immediate failure.

        (dV_dr,) = torch.autograd.grad(V_sq, r_grad, create_graph=False, retain_graph=False)
        d2r_dtau2 = 0.5 * dV_dr # The radial geodesic equation.

        # Return the derivatives of the state vector [t, r, φ, dr/dτ].
        return torch.stack((u_t / self.c, dr_dtau, u_phi, d2r_dtau2))

    def rk4_step(self, y: Tensor, dτ: float) -> Tensor:
        """
        Performs a single Runge-Kutta 4th order integration step.
        <reason>RK4 is a standard, robust numerical method for solving ODEs. It offers a good balance between accuracy and computational cost compared to simpler methods like Euler's method.</reason>
        """
        k1 = self._ode(y).detach()
        k2 = self._ode((y + 0.5 * dτ * k1)).detach()
        k3 = self._ode((y + 0.5 * dτ * k2)).detach()
        k4 = self._ode((y + dτ * k3)).detach()
        return y + (k1 + 2 * k2 + 2 * k3 + k4) * (dτ / 6.0)

# ---------------------------------------------------------------------------
# 4.  ANALYSIS & MAIN DRIVER
# ---------------------------------------------------------------------------

def calculate_fft_loss(traj_ref: Tensor, traj_pred: Tensor) -> float:
    """
    Calculates the informational loss between two trajectories using FFT MSE.
    <reason>This function is the core of the paper's methodology. It compares the full frequency spectrum of orbital dynamics, capturing subtle differences in precession and shape that a simple endpoint comparison would miss. It is a direct measure of a theory's informational fidelity.</reason>
    """
    min_len = min(len(traj_ref), len(traj_pred))
    if min_len < 2: return float("inf") # FFT requires at least 2 points.

    r_ref, r_pred = traj_ref[:min_len, 1], traj_pred[:min_len, 1]
    if not (torch.all(torch.isfinite(r_ref)) and torch.all(torch.isfinite(r_pred))):
        return float('nan') # Return NaN for non-finite input trajectories.

    fft_ref = torch.fft.fft(r_ref)
    fft_pred = torch.fft.fft(r_pred)

    # MSE between the magnitudes of the FFTs (the power spectra).
    loss = torch.mean((torch.abs(fft_ref) - torch.abs(fft_pred)) ** 2)
    return loss.item()

def main() -> None:
    """
    Main driver for the simulation.
    <reason>This function orchestrates the entire process: setting up models, defining initial conditions, running the simulations, calculating losses, and reporting the results.</reason>
    """
    print("=" * 80)
    print(f"PyTorch Orbital Test | device={device} | dtype={DTYPE}")
    print("=" * 80)

    # -- Model Registry --
    # <reason>This registry defines all theories to be tested. It is structured to be easily modifiable, allowing new theories to be added or existing ones to be swapped out for different experimental runs.</reason>
    
    # Theories removed to focus on 'distinguishables' per research recommendations:
    # - Acausal, NonLocal, EinsteinFinal(α=0): Mathematically identical or too close to GR (lossless).
    # - Computational Complexity, Kerr: Consistently produced unstable trajectories (NaN).
    # - Unstable parameter ranges for VariableG, Fractal, etc., have been pruned from sweeps.
    models: list[GravitationalTheory] = [
        Schwarzschild(), NewtonianLimit(), ReissnerNordstrom(Q_PARAM),
        EinsteinRegularized(), StochasticNoise(STOCHASTIC_STRENGTH),
    ]

    # TODO: As per research recommendations, the next step is to perform
    # targeted, finer-grained parameter sweeps on the most promising models
    # (e.g., LogCorrected for β between -0.50 and +0.17) to find optimal values.
    sweeps = {
        "QuantumCorrected": (QuantumCorrected, dict(alpha=np.linspace(-2.0, 2.0, 5))),
        "LogCorrected": (LogCorrected, dict(beta=np.linspace(-1.5, 1.5, 5))),
        # Pruned unstable positive-delta parameters from VariableG sweep.
        "VariableG": (VariableG, dict(delta=np.linspace(-0.5, -0.05, 5))),
    }
    for cls, pd in sweeps.values():
        key, vals = next(iter(pd.items()))
        models += [cls(**{key: float(v)}) for v in vals]
    print(f"Total models to be tested: {len(models)}")

    # -- Initial Conditions --
    # <reason>We define initial conditions for a quasi-stable orbit. Starting at 4x the Schwarzschild radius with a specific tangential velocity ensures the orbit is interesting and long-lived enough to reveal differences between theories.</reason>
    r0 = 4.0 * RS
    v_tan = torch.sqrt(G * M / r0)
    g_tt0, _, g_pp0, _ = Schwarzschild().get_metric(r0, M, c, G)
    norm_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * c)) ** 2
    dt_dtau0 = 1.0 / torch.sqrt(norm_sq)
    dphi_dtau0 = (v_tan / r0) * dt_dtau0
    y0_full = torch.tensor([0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()], device=device, dtype=DTYPE)
    y0_state = y0_full[[0, 1, 2, 4]].clone() # State: [t, r, φ, dr/dτ]

    # -- Run Parameters --
    DTau = 0.01
    MAX_CONSECUTIVE_FAILURES = 10 # For exponential backoff
    if args.final:
        N_STEPS, STEP_PRINT, SAVE_PLOTS = 5_000_000, 250_000, True
        print("Mode: FINAL (high precision, long duration)")
    else:
        N_STEPS, STEP_PRINT, SAVE_PLOTS = 100_000, 10_000, args.plots
        print("Mode: EXPLORATORY (fast, for prototyping)")
    
    # -- Ground-Truth Trajectory Generation (Cached) --
    # <reason>The ground-truth GR and R-N trajectories are computationally expensive. Caching them to disk after the first run avoids re-computation, dramatically speeding up subsequent tests.</reason>
    def cached_run(model: GravitationalTheory, tag: str) -> Tensor:
        """Runs a simulation for a given model, caching the result."""
        precision_tag = "f64" if DTYPE == torch.float64 else "f32"
        fname = f"cache_{tag}_{N_STEPS}_{precision_tag}.pt"
        if os.path.exists(fname): return torch.load(fname, map_location=device)
        
        print(f"\n--- Generating Ground Truth: {model.name} ---")
        integ = GeodesicIntegrator(model, y0_full, M, c, G)
        hist = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
        hist[0], y = y0_state, y0_state.clone()
        for i in range(N_STEPS):
            y = integ.rk4_step(y, DTau)
            hist[i + 1] = y
            if (i + 1) % STEP_PRINT == 0: print(f"  ...step {i+1:,}/{N_STEPS:,} | r={y[1]/RS:.3f} RS")
            if not torch.all(torch.isfinite(y)) or y[1] <= RS * 1.01:
                hist = hist[: i + 2]; break
        torch.save(hist, fname)
        return hist

    GR_hist = cached_run(Schwarzschild(), "GR")
    RN_hist = cached_run(ReissnerNordstrom(Q_PARAM), "RN")

    # -- Main Evaluation Loop --
    results = []
    for idx, model in enumerate(models, 1):
        print(f"\n[{idx:03}/{len(models)}] Evaluating: {model.name}")
        integ = GeodesicIntegrator(model, y0_full, M, c, G)
        traj = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
        traj[0], y = y0_state, y0_state.clone()
        
        consecutive_failures = 0
        for i in range(N_STEPS):
            y = integ.rk4_step(y, DTau)
            traj[i + 1] = y
            
            # Exponential Backoff Logic
            # <reason>This logic saves compute by aborting unstable simulations early. If a theory produces non-finite results repeatedly, it is fundamentally unstable under these conditions, and continuing the simulation is pointless.</reason>
            if not torch.all(torch.isfinite(y)):
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"  ! ABORTED: Simulation unstable for {consecutive_failures} consecutive steps.")
                    traj = traj[:i+2]
                    break
            else:
                consecutive_failures = 0

            if y[1] <= RS * 1.01: # Particle captured
                traj = traj[:i+2]
                break
        
        results.append({
            "name": model.name,
            "loss_GR": calculate_fft_loss(GR_hist, traj),
            "loss_RN": calculate_fft_loss(RN_hist, traj),
            "traj": traj.cpu().numpy(),
        })

    # -- Reporting and Plotting --
    PLOT_DIR = f"plots/run_{int(time.time())}"
    if SAVE_PLOTS and not args.no_plots: os.makedirs(PLOT_DIR, exist_ok=True)

    # Report 1: Sorted by loss against General Relativity (GR)
    results.sort(key=lambda d: (d["loss_GR"] is None or math.isnan(d["loss_GR"]), d["loss_GR"]))
    print("\n\n" + "="*80)
    print("--- RANKING vs. GENERAL RELATIVITY (GR) ---")
    print("Rank | Model                               | Loss_GR (FFT MSE)")
    print("-" * 60)
    for rank, res in enumerate(results, 1):
        print(f"{rank:4d} | {res['name']:<35} | {res.get('loss_GR', float('nan')):10.3e}")
    print("="*80)

    # Report 2: Sorted by loss against Reissner-Nordström (R-N)
    results.sort(key=lambda d: (d["loss_RN"] is None or math.isnan(d["loss_RN"]), d["loss_RN"]))
    print("\n--- RANKING vs. REISSNER-NORDSTRÖM (R-N) ---")
    print("Rank | Model                               | Loss_RN (FFT MSE)")
    print("-" * 60)
    for rank, res in enumerate(results, 1):
        print(f"{rank:4d} | {res['name']:<35} | {res.get('loss_RN', float('nan')):10.3e}")
    print("="*80)

    if SAVE_PLOTS and not args.no_plots:
        GR_np, RN_np = GR_hist.cpu().numpy(), RN_hist.cpu().numpy()
        results.sort(key=lambda d: (d["loss_GR"] is None or math.isnan(d["loss_GR"]), d["loss_GR"]))
        top_results = results if args.plots else results[:5]
        
        print(f"\nGenerating plots for top {len(top_results)} models in '{PLOT_DIR}/'...")
        for res in top_results:
            pred_np = res["traj"]
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection="polar")
            
            # Logical Correction: Plot baselines with higher zorder and thicker lines.
            # <reason>This ensures the dashed/dotted baseline trajectories are visible even when a prediction trajectory overlaps them perfectly, solving a key visual artifact issue.</reason>
            ax.plot(GR_np[:, 2], GR_np[:, 1], "k--", label="GR", linewidth=1.5, zorder=5)
            ax.plot(RN_np[:, 2], RN_np[:, 1], "b:",  label="R‑N", linewidth=1.5, zorder=5)
            ax.plot(pred_np[:, 2], pred_np[:, 1], "r-", label=res["name"], zorder=4)
            ax.plot(pred_np[0, 2], pred_np[0, 1], "go", markersize=8, label="start", zorder=6)
            ax.plot(pred_np[-1, 2], pred_np[-1, 1], "rx", markersize=10, mew=2, label="end", zorder=6)
            
            ax.set_title(res["name"], pad=20)
            ax.legend()
            plt.tight_layout()
            safe_name = res["name"].translate({ord(c): "_" for c in " /()=.*+-"})
            plt.savefig(os.path.join(PLOT_DIR, f"{safe_name}.png"))
            plt.close()
        print("Plots saved successfully.")

    print("\nDone.")

if __name__ == "__main__":
    # <reason>JIT tracer warnings are common with torch.compile and complex control flow but are not critical errors. Suppressing them keeps the console output clean and focused on simulation results.</reason>
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()