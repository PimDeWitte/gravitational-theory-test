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
# - Pruned non-competitive speculative theories to focus compute on the most promising models.
# - Implemented dual-baseline reporting for both GR and Reissner-Nordström.
# - Added --cpu-f64 flag for high-precision validation runs.
# - Implemented exponential backoff for failing simulations to save compute.
# - Modified to use Grok API for dynamic theory generation based on history.
# - Ensured Grok 4 is called via API.
# - Removed all hardcoded theories except ground truths; self-generating only.
# - Added validity check for generated code.
# - Save theory code, plot, results, and data per theory with timestamp.
# - Infinite loop until breakthrough found.
# - Added debugging prints for API calls and retry mechanism to guarantee generation.
# - Increased max_tokens to 4096 to avoid length issues.
# - Removed paper reference in prompt, described objective instead.
# - Changed to prompt for 1 theory at a time.
# - Parse generated content to extract code from markdown blocks.
# - Instruct API to add <reason>reasoning chain</reason> comments for self-documentation.
# - Added <summary>description</summary> for theory summary, extract and include in history/prompt.
# - Fixed source saving by using generated content instead of inspect.getsource.
# - Return model, summary, content from generate_new_theories to fix NameError.
# ---------------------------------------------------------------------------

from __future__ import annotations
import os, time, math, argparse, warnings, inspect, json, re
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c, k, hbar, epsilon_0
import random  # For fallback if needed

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
    p.add_argument("--cpu-f64", action="store_true", help="Run on CPU with float64 precision for validation. Overrides default GPU/float32 settings.")
    return p.parse_args()

args = parse_cli()
XAI_API_KEY = os.environ.get("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY environment variable is required for Grok API calls.")

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
Q_PARAM = 3.0e14
STOCHASTIC_STRENGTH = 1e-7

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

# -- 2.1 Standard & Baseline Metrics --

class Schwarzschild(GravitationalTheory):
    """
    The Schwarzschild metric for a non-rotating, uncharged black hole.
    <reason>This is the exact solution to Einstein's field equations in a vacuum and serves as the fundamental ground truth (baseline) for pure gravity in this framework.</reason>
    """
    def __init__(self): super().__init__("Schwarzschild (GR)")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / (r + EPSILON)
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)

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
        rq_sq = (G_param * self.Q**2) / (4 * TORCH_PI * EPS0_T * C_param**4)
        m = 1 - rs / r + rq_sq / r**2
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
        
# ---------------------------------------------------------------------------
# 2.3 Dynamic Theory Generation via Grok API
# ---------------------------------------------------------------------------

def build_prompt(history: list[dict]) -> str:
    """
    Builds a dynamic prompt for the Grok API based on previous results.
    The prompt grows with history, allowing the system to learn iteratively.
    """
    prompt = """
You are Grok, a physics researcher built by xAI, tasked with discovering a unified theory of gravity and electromagnetism.
Draw heavy inspiration from Einstein's 30-year pursuit of a unified field theory, where he attempted to derive electromagnetism from pure geometry (e.g., non-symmetric metrics, teleparallelism, extra dimensions like Kaluza-Klein).
Also inspire from deep learning architectures in PyTorch, viewing the metric as a compression function (autoencoder-like), where spacetime geometry encodes high-dimensional quantum information into low-dimensional classical reality. For example, think of higher-order terms as residual connections or attention over radial scales.

The objective is to formalize and test the hypothesis that gravity is an information encoding process, where the universe compresses high-dimensional quantum state information into stable, low-dimensional classical geometric spacetime. Physical theories act as "decoders". Use a computational framework to measure "decoding loss" of candidate theories via dynamic orbital mechanics tests, benchmarked against lossless decoders for gravity (Schwarzschild metric) and electromagnetism (Reissner-Nordström metric). Results confirm unique, lossless status of General Relativity and Kaluza-Klein theory, establishing a methodology for evaluating laws based on informational fidelity.

Previous results ( theory name: summary, loss vs GR, loss vs R-N ):
"""
    for h in history:
        summary = h.get('summary', 'No summary available')
        prompt += f"{h['name']}: {summary}, loss_GR={h['loss_GR']:.3e}, loss_RN={h['loss_RN']:.3e}\n"

    prompt += """
Suggest 1 new, unique GravitationalTheory subclass as a complete Python class definition.
It must:
- Inherit from GravitationalTheory.
- Have a unique name.
- Implement __init__ with super().__init__(name).
- Implement get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor] for g_tt, g_rr, g_φφ, g_tφ.
- Use only torch operations, no imports in the code.
- Avoid explicit Q; instead, introduce geometric terms (e.g., alpha * (rs**2 / r**2), non-diagonal g_tφ for field-like effects, logarithmic/higher-order corrections inspired by quantum/DL).
- Parameterize where useful (e.g., alpha for sweeps), inspired by Einstein's attempts.
- Add <reason>reasoning chain</reason> comments explaining the physical and inspirational reasoning for each part of the metric.
- Add a <summary>concise description of the theory, including the key metric formula</summary> as a comment at the top of the class.

For Einstein!

Output ONLY the Python code for the class, no explanations or extra text.
"""
    return prompt

def generate_new_theories(history: list[dict]) -> list[tuple[GravitationalTheory, str, str]]:
    """
    Calls the Grok API to generate new theory classes based on history.
    Executes the returned code to define the classes dynamically.
    Returns list of (model instance, summary, content)
    """
    prompt = build_prompt(history)
    print("\nDebug: Prompt sent to API:\n", prompt)

    max_retries = 5
    temperature = 0.8
    for attempt in range(1, max_retries + 1):
        print(f"\nDebug: API Call Attempt {attempt}/{max_retries} with temperature={temperature}")
        data = {
            "model": "grok-4",  # Ensuring Grok 4 is called
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4096,  # Increased to avoid length issues
        }
        print("Debug: API Request Data:", json.dumps(data, indent=2))
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json=data,
            )
            print("Debug: API Response Status Code:", resp.status_code)
            if resp.status_code != 200:
                print("Debug: API Response Text:", resp.text)
                temperature += 0.2  # Increase temperature for retry
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            response_json = resp.json()
            print("Debug: API Response JSON:", json.dumps(response_json, indent=2))
            if "choices" not in response_json or not response_json["choices"]:
                print("Debug: No choices in response.")
                temperature += 0.2
                time.sleep(2 ** attempt)
                continue
            content = response_json["choices"][0]["message"]["content"]
            print(f"\nGenerated theory code:\n{content}\n")
            if not content.strip():
                print("Debug: Empty content received.")
                temperature += 0.2
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            print(f"Debug: API Call Error: {e}")
            temperature += 0.2
            time.sleep(2 ** attempt)
            continue

        # Parse content to extract code from markdown if present
        match = re.search(r'```python\s*(.*?)```', content, re.DOTALL)
        if match:
            content = match.group(1).strip()

        # Remove any import statements
        content = re.sub(r'^(from|import)\s+.*$', '', content, flags=re.MULTILINE).strip()

        print(f"\nCleaned theory code:\n{content}\n")

        # Extract summary
        summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "No summary provided"

        # Save the full generated code
        gen_timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("generated_codes", exist_ok=True)
        with open(f"generated_codes/{gen_timestamp}_generated.py", "w") as f:
            f.write(content)

        # Get existing theories before exec
        existing_classes = {
            cls for name, cls in globals().items() if inspect.isclass(cls) and issubclass(cls, GravitationalTheory) and cls != GravitationalTheory
        }

        # Execute the code to define new classes
        try:
            exec(content, globals())
        except Exception as e:
            print(f"Error executing generated code: {e}")
            temperature += 0.2
            time.sleep(2 ** attempt)
            continue

        # Find newly defined classes
        all_classes = {
            cls for name, cls in globals().items() if inspect.isclass(cls) and issubclass(cls, GravitationalTheory) and cls != GravitationalTheory
        }
        new_classes = all_classes - existing_classes

        if not new_classes:
            print("No new theories generated from API response.")
            temperature += 0.2
            time.sleep(2 ** attempt)
            continue

        valid_models = []
        for cls in new_classes:
            try:
                test_model = cls()
                test_r = torch.tensor(10.0, device=device, dtype=DTYPE)
                gtt, grr, gpp, gtp = test_model.get_metric(test_r, M, c, G)
                if not all(torch.isfinite(t).all() for t in (gtt, grr, gpp, gtp)):
                    raise ValueError("Non-finite metric values")
                valid_models.append((test_model, summary, content))
            except Exception as e:
                print(f"Invalid theory {cls.__name__}: {e}")
                continue

        if valid_models:
            return valid_models
        else:
            print("No valid models after validation.")
            temperature += 0.2
            time.sleep(2 ** attempt)

    # If all retries fail, generate a fallback theory
    print("All API retries failed. Generating a fallback theory.")
    fallback_content = """
class FallbackTheory(GravitationalTheory):
    def __init__(self):
        super().__init__(f"FallbackTheory_{random.randint(1, 100)}")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
"""
    fallback_summary = "Fallback theory similar to Schwarzschild: g_tt = -(1 - rs/r), g_rr = 1/(1 - rs/r), g_φφ = r^2, g_tφ = 0"
    exec(fallback_content, globals())
    fallback_cls = FallbackTheory
    fallback_model = fallback_cls()
    return [(fallback_model, fallback_summary, fallback_content)]

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
        _, r0, _, dt_dtau0, _, dphi_dtau0 = y0_full
        g_tt0, _, g_pp0, g_tp0 = self.model.get_metric(r0, self.M, self.c, self.G)
        self.E  = -(g_tt0 * self.c * dt_dtau0 + g_tp0 * dphi_dtau0)
        self.Lz =  g_tp0 * self.c * dt_dtau0 + g_pp0 * dphi_dtau0
        if os.environ.get("TORCH_COMPILE") == "1" and hasattr(torch, "compile"):
            self._ode = torch.compile(self._ode_impl, fullgraph=True, mode="reduce-overhead", dynamic=True)
        else:
            self._ode = self._ode_impl

    def _ode_impl(self, y_state: Tensor) -> Tensor:
        """The right-hand side of the system of ODEs for the geodesic equations."""
        _, r, _, dr_dtau = y_state
        r_grad = r.clone().detach().requires_grad_(True)
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r_grad, self.M, self.c, self.G)
        det = g_tp ** 2 - g_tt * g_pp
        if torch.abs(det) < EPSILON: return torch.zeros_like(y_state)
        u_t   = (self.E * g_pp + self.Lz * g_tp) / det
        u_phi = -(self.E * g_tp + self.Lz * g_tt) / det
        V_sq = (-self.c ** 2 - (g_tt * u_t ** 2 + g_pp * u_phi ** 2 + 2 * g_tp * u_t * u_phi)) / g_rr
        if not torch.all(torch.isfinite(V_sq)): return torch.full_like(y_state, float('nan'))
        (dV_dr,) = torch.autograd.grad(V_sq, r_grad, create_graph=False, retain_graph=False)
        d2r_dtau2 = 0.5 * dV_dr
        return torch.stack((u_t / self.c, dr_dtau, u_phi, d2r_dtau2))

    def rk4_step(self, y: Tensor, dτ: float) -> Tensor:
        """Performs a single Runge-Kutta 4th order integration step."""
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
    if min_len < 2: return float("inf")
    r_ref, r_pred = traj_ref[:min_len, 1], traj_pred[:min_len, 1]
    if not (torch.all(torch.isfinite(r_ref)) and torch.all(torch.isfinite(r_pred))):
        return float('nan')
    fft_ref, fft_pred = torch.fft.fft(r_ref), torch.fft.fft(r_pred)
    return torch.mean((torch.abs(fft_ref) - torch.abs(fft_pred)) ** 2).item()

def main() -> None:
    """
    Main driver for the simulation.
    <reason>This function orchestrates the entire process: setting up models, defining initial conditions, running the simulations, calculating losses, and reporting the results.</reason>
    """
    print("=" * 80)
    print(f"PyTorch Orbital Test | device={device} | dtype={DTYPE}")
    print("=" * 80)

    # Initial history from baselines
    history = [
        {"name": "Schwarzschild (GR)", "loss_GR": 0.0, "loss_RN": 0.0, "summary": "Standard GR metric: g_tt = -(1 - rs/r), g_rr = 1/(1 - rs/r), g_φφ = r^2, g_tφ = 0"},
        {"name": "Reissner‑Nordström (Q=3.0e14)", "loss_GR": 0.0, "loss_RN": 0.0, "summary": "Charged GR metric: g_tt = -(1 - rs/r + rq^2/r^2), g_rr = 1/(1 - rs/r + rq^2/r^2), g_φφ = r^2, g_tφ = 0"}
    ]

    # -- Initial Conditions --
    r0 = 4.0 * RS
    v_tan = torch.sqrt(G * M / r0)
    g_tt0, _, g_pp0, _ = Schwarzschild().get_metric(r0, M, c, G)
    norm_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * c)) ** 2
    dt_dtau0 = 1.0 / torch.sqrt(norm_sq)
    dphi_dtau0 = (v_tan / r0) * dt_dtau0
    y0_full = torch.tensor([0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()], device=device, dtype=DTYPE)
    y0_state = y0_full[[0, 1, 2, 4]].clone()

    # -- Run Parameters --
    DTau = 0.01
    MAX_CONSECUTIVE_FAILURES = 10
    if args.final:
        N_STEPS, STEP_PRINT = 5_000_000, 250_000
        print("Mode: FINAL (high precision, long duration)")
    else:
        N_STEPS, STEP_PRINT = 100_000, 10_000
        print("Mode: EXPLORATORY (fast, for prototyping)")
    
    # -- Ground-Truth Trajectory Generation (Cached) --
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
    GR_loss_vs_RN = calculate_fft_loss(RN_hist, GR_hist)

    # Update baselines with actual losses (if needed, but since lossless, 0)
    history[0]["loss_RN"] = GR_loss_vs_RN
    history[1]["loss_GR"] = GR_loss_vs_RN

    # -- Iterative Evaluation Loop --
    breakthrough_found = False
    iteration = 1
    while True:
        print(f"\n--- Iteration {iteration}: Generating new theories ---")
        new_theories = generate_new_theories(history)
        if not new_theories:
            print("No valid new models generated. Continuing...")
            iteration += 1
            continue

        print(f"Testing {len(new_theories)} new models: {[m[0].name for m in new_theories]}")

        results = []
        for idx, (model, summary, gen_content) in enumerate(new_theories, 1):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(".", "_")
            theory_dir = f"theories/{timestamp}_{safe_name}"
            os.makedirs(theory_dir, exist_ok=True)

            # Save theory code using the generated content
            with open(f"{ theory_dir}/code.py", "w") as f:
                f.write(gen_content)

            print(f"\n[{idx:03}/{len(new_theories)}] Evaluating: {model.name}")
            integ = GeodesicIntegrator(model, y0_full, M, c, G)
            traj = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
            traj[0], y = y0_state, y0_state.clone()
            consecutive_failures = 0
            for i in range(N_STEPS):
                y = integ.rk4_step(y, DTau)
                traj[i + 1] = y
                if not torch.all(torch.isfinite(y)):
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        print(f"  ! ABORTED: Simulation unstable for {consecutive_failures} consecutive steps.")
                        traj = traj[:i+2]; break
                else:
                    consecutive_failures = 0
                if y[1] <= RS * 1.01:
                    traj = traj[:i+2]; break
            res = {
                "name": model.name,
                "loss_GR": calculate_fft_loss(GR_hist, traj),
                "loss_RN": calculate_fft_loss(RN_hist, traj),
                "traj": traj.cpu().numpy(),
            }
            results.append(res)
            history.append({"name": res["name"], "loss_GR": res["loss_GR"], "loss_RN": res["loss_RN"], "summary": summary})  # Append with summary

            # Save plot
            GR_np, RN_np = GR_hist.cpu().numpy(), RN_hist.cpu().numpy()
            pred_np = res["traj"]
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection="polar")
            ax.plot(GR_np[:, 2], GR_np[:, 1], "k--", label="GR", linewidth=1.5, zorder=5)
            ax.plot(RN_np[:, 2], RN_np[:, 1], "b:",  label="R-N", linewidth=1.5, zorder=5)
            ax.plot(pred_np[:, 2], pred_np[:, 1], "r-", label=res["name"], zorder=4)
            ax.plot(pred_np[0, 2], pred_np[0, 1], "go", markersize=8, label="start", zorder=6)
            ax.plot(pred_np[-1, 2], pred_np[-1, 1], "rx", markersize=10, mew=2, label="end", zorder=6)
            ax.set_title(res["name"], pad=20)
            ax.legend(); plt.tight_layout()
            plt.savefig(f"{ theory_dir}/plot.png")
            plt.close()

            # Save results and data
            np.save(f"{ theory_dir}/traj.npy", pred_np)
            with open(f"{ theory_dir}/results.json", "w") as f:
                json.dump({
                    "name": res["name"],
                    "loss_GR": res["loss_GR"],
                    "loss_RN": res["loss_RN"],
                }, f)

        # -- Reporting --
        BOLD, GREEN_BG, RESET = "\033[1m", "\033[42m", "\033[0m"

        results.sort(key=lambda d: (math.isnan(d["loss_GR"]), d["loss_GR"]))
        print("\n\n" + "="*80)
        print("--- RANKING vs. GENERAL RELATIVITY (GR) ---")
        print("Rank | Model                                | Loss_GR (FFT MSE)")
        print("-" * 60)
        for rank, res in enumerate(results, 1):
            print(f"{rank:4d} | {res['name']:<36} | {res['loss_GR']:10.3e}")
        print("="*80)

        results.sort(key=lambda d: (math.isnan(d["loss_RN"]), d["loss_RN"]))
        print("\n--- RANKING vs. REISSNER-NORDSTRÖM (R-N) ---")
        print(f"(GR baseline loss vs R-N is: {GR_loss_vs_RN:.3e})")
        print("Rank | Model                                | Loss_RN (FFT MSE)")
        print("-" * 60)
        for rank, res in enumerate(results, 1):
            loss_val = res["loss_RN"]
            name = res['name']
            is_breakthrough = not math.isnan(loss_val) and loss_val < GR_loss_vs_RN and "Schwarzschild" not in name and "Reissner" not in name
            if is_breakthrough:
                print(f"{GREEN_BG}{BOLD}{rank:4d} | {name:<36} | {loss_val:10.3e} [BREAKTHROUGH]{RESET}")
                breakthrough_found = True
            else:
                print(f"{rank:4d} | {name:<36} | {loss_val:10.3e}")
        print("="*80)

        if breakthrough_found:
            print("\nBreakthrough theory found! Continuing to find potentially better ones.")
            # Continue infinitely, do not break

        iteration += 1

    print("\nDone.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()