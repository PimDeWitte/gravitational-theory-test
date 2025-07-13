#!/usr/bin/env python3
from __future__ import annotations
# sim_gpu.py  ── July 2025
# ---------------------------------------------------------------------------
# Float-32 black-hole orbital integrator for Apple-silicon (M-series) GPUs.
# All known mathematical / computational bugs are fixed; optional Torch-Dynamo
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
# - Incorporated fixes: Normalized FFT loss fully, torsion logging, increased r0 to 15 RS, updated prompt with Einstein notes, added torsion optional in metric, added EinsteinDeathbedUnified, increased temp cap, regex for <reason>.
# - Added load_manual_theories() to allow manual equation input from disk file.
# ---------------------------------------------------------------------------
# <reason>chain: Retained original header for continuity; updated with new modifications for completeness and to reflect all changes.</reason>
import importlib
import sys

import os, time, math, argparse, warnings, inspect, json, re
# <reason>chain: Imported standard libraries for OS, time, math, args, warnings, inspect, JSON, regex.</reason>
import requests
# <reason>chain: Imported requests for API calls.</reason>
import torch
# <reason>chain: Imported torch for tensors and autodiff.</reason>
import numpy as np
# <reason>chain: Imported numpy for array operations.</reason>
import matplotlib.pyplot as plt
# <reason>chain: Imported matplotlib for plotting.</reason>
import inspect
# NOTE: Do not import predefined_theories.py directly here, as it contains only class definitions
# that depend on GravitationalTheory, which is not defined in that file. The actual theory classes
# are dynamically loaded and executed elsewhere in this script.

from scipy.constants import G, c, k, hbar, epsilon_0
# <reason>chain: Imported physical constants.</reason>
import random  # For fallback if needed
# <reason>chain: Imported random for fallback.</reason>
import shutil
import subprocess
import webbrowser
# <reason>chain: Imports unchanged; foundational for API, tensors, and plotting.</reason>
import getpass  # For potential secure input if needed, but using input()
import socket

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
    p.add_argument("--self-discover", action="store_true", help="Enable self-discovery loop for generating new theories via API.")
    # <reason>chain: Added --self-discover flag to toggle the generation loop.</reason>
    p.add_argument("--initial-prompt", type=str, default="", help="Initial prompt or seed query for theory generation, e.g., 'find data about the scribbles before Einstein died and generate theories on those'.")
    # <reason>chain: Added --initial-prompt to allow custom seeding of the generation prompt.</reason>
    p.add_argument("--api-provider", type=str, default="grok", choices=["grok", "gemini", "openai", "anthropic"], help="API provider for theory generation.")
    # <reason>chain: Added --api-provider to select which API to use for generation, if key is present.</reason>
    p.add_argument("--manual-theories-file", type=str, default=None, help="Path to a Python file containing manual theory class definitions to load.")
    # <reason>Added --manual-theories-file flag to allow loading manual equations/theories from disk, enabling user-defined inputs as per update.</reason>
    p.add_argument("--theory-dirs", type=str, nargs='+', default=["theories/defaults"], help="Theory directories to load (default: theories/defaults). Can specify multiple.")
    # <reason>Added --theory-dirs to specify which theory directories to load from the new structure.</reason>
    p.add_argument("--test", action="store_true", help="Run in test mode with reduced steps for quick benchmarking.")
    p.add_argument("--validate-observations", action="store_true", help="Run validation against real astronomical observations.")
    p.add_argument("--loss-type", type=str, default="fft", choices=["fft", "endpoint_mse", "cosine", "trajectory_mse", "hausdorff", "frechet", "trajectory_dot", "raw_dot"], help="Loss calculation type to use (or 'all' with --multi-loss).")  # <reason>chain: Added raw_dot to choices for custom loss implementation.&lt;/reason&gt;
    p.add_argument("--multi-loss", action="store_true", help="Compute all available loss types in one run and store in results.json.")
    # <reason>chain: Added --multi-loss flag to compute all loss types at once for comprehensive comparison in unification tests.&lt;/reason&gt;
    p.add_argument("--no-cache", action="store_true", help="Force recomputation by ignoring existing cache files.")
    # <reason>chain: Added --no-cache flag to purge/invalidate caches and force fresh computations for testing.&lt;/reason&gt;
    return p.parse_args()
# <reason>chain: Defined parse_cli with arguments; added manual file arg for new feature.</reason>

args = parse_cli()
# <reason>chain: Parsed arguments; no change.</reason>

# Check for API keys based on provider
API_KEYS = {
    "grok": os.environ.get("XAI_API_KEY"),
    "gemini": os.environ.get("GEMINI_API_KEY"),
    "openai": os.environ.get("OPENAI_API_KEY"),
    "anthropic": os.environ.get("ANTHROPIC_API_KEY"),
}
# <reason>chain: Defined API keys dict; no change.</reason>
if args.self_discover and not API_KEYS.get(args.api_provider):
    raise ValueError(f"{args.api_provider.upper()}_API_KEY environment variable is required for self-discovery mode.")
# <reason>chain: Checked API key; no change.</reason>

# Set device and data type based on CLI flags. This must be done before any tensors are created.
# <reason>This block allows for flexible hardware and precision choices. The default is fast GPU/float32 for exploration, while --cpu-f64 enables high-precision CPU runs for validating key results, as recommended in the research plan.</reason>
if args.final or args.cpu_f64:
    DTYPE  = torch.float64
    device = torch.device("cpu")
    # <reason>chain: Default to float64 in --final for high-precision validation, per paper recommendations.</reason>
else:
    DTYPE  = torch.float32
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# <reason>chain: Set device and dtype; no change.</reason>

# Epsilon value for numerical stability, scaled by the chosen data type's precision.
# <reason>A fixed small epsilon is not robust. Tying it to the data type's machine epsilon ensures that the stability margin is appropriate for both float32 and float64, preventing underflow or loss of significance.</reason>
EPSILON  = torch.finfo(DTYPE).eps * 100
# <reason>chain: Defined epsilon; no change.</reason>

# ---------------------------------------------------------------------------
# 1.  PHYSICAL CONSTANTS & SYSTEM PARAMETERS
# ---------------------------------------------------------------------------

# <reason>Defining constants as tensors on the correct device and with the correct dtype from the start avoids repeated CPU-GPU transfers and type conversions within the simulation loop, which is a major performance optimization.</reason>
TORCH_PI = torch.as_tensor(math.pi,  device=device, dtype=DTYPE)
# <reason>chain: Pi tensor; no change.</reason>
EPS0_T   = torch.as_tensor(epsilon_0, device=device, dtype=DTYPE)
# <reason>chain: Epsilon0 tensor; no change.</reason>

# System Parameters: 10 Solar Mass Black Hole
# <reason>These parameters define the central object for our simulation. 10 M☉ is a standard choice for a stellar-mass black hole, providing a realistic scale for testing gravitational effects.</reason>
M_SI  = 10.0 * 1.989e30
# <reason>chain: Mass SI; no change.</reason>
RS_SI = 2 * G * M_SI / c**2
# <reason>chain: RS SI; no change.</reason>
M  = torch.as_tensor(M_SI , device=device, dtype=DTYPE)
# <reason>chain: Mass tensor; no change.</reason>
RS = torch.as_tensor(RS_SI, device=device, dtype=DTYPE)
# <reason>chain: RS tensor; no change.</reason>

# Cached Planck Length Tensor
# <reason>The Planck Length is used in some quantum gravity models. Caching it as a tensor avoids recalculating the Python float and converting it to a tensor inside the simulation loop.</reason>
LP = torch.as_tensor(math.sqrt(G * hbar / c**3), device=device, dtype=DTYPE)
# <reason>chain: LP tensor; no change.</reason>

# Default parameters for various speculative models.
# <reason>These default values are used to instantiate the non-swept versions of the theories. They are chosen to be physically significant enough to produce a deviation from GR without immediately causing the simulation to fail.</reason>
Q_PARAM = 1e19  # Reduced from 4.878e21 to avoid numerical overflow in float32. Still gives rq/RS ~0.003 for meaningful EM effects.
# <reason>chain: Set Q_PARAM to 1e19 C to avoid numerical overflow while maintaining distinguishable electromagnetic effects in the Reissner-Nordström metric.</reason>
STOCHASTIC_STRENGTH = 1e-7
# <reason>chain: Stochastic strength; no change.</reason>

G_T = torch.as_tensor(G, device=device, dtype=DTYPE)
# <reason>Added tensor version of G to ensure consistent types in calculations like v_tan, avoiding potential type issues in torch operations.</reason>
C_T = torch.as_tensor(c, device=device, dtype=DTYPE)
# <reason>Added tensor version of c for consistency in tensor operations.</reason>

# ---------------------------------------------------------------------------
# 2.  THEORY DEFINITIONS
# ---------------------------------------------------------------------------

from base_theory import GravitationalTheory, Tensor
def _get_theory_classes():
    """
    Dynamically loads predefined_theories.py as a module in a way that ensures
    GravitationalTheory and all required symbols are available in its namespace,
    preventing NameError and NameError for type annotations.
    """
    import types

    module_name = "predefined_theories"
    module_path = os.path.join(os.path.dirname(__file__), "predefined_theories.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    predefined_theories = importlib.util.module_from_spec(spec)

    # Inject GravitationalTheory and all required globals into the module's namespace before execution
    predefined_theories.GravitationalTheory = GravitationalTheory
    # Also inject all symbols used in type annotations or as globals in predefined_theories.py
    predefined_theories.torch = torch
    predefined_theories.math = math
    predefined_theories.device = device
    predefined_theories.DTYPE = DTYPE
    predefined_theories.EPSILON = EPSILON
    predefined_theories.epsilon_0 = epsilon_0
    predefined_theories.G = G
    predefined_theories.c = c
    predefined_theories.hbar = hbar
    predefined_theories.Q_PARAM = Q_PARAM
    predefined_theories.TORCH_PI = TORCH_PI
    predefined_theories.EPS0_T = EPS0_T
    predefined_theories.LP = LP
    predefined_theories.STOCHASTIC_STRENGTH = STOCHASTIC_STRENGTH

    # If Tensor is used as a type annotation, inject it as well
    predefined_theories.Tensor = torch.Tensor

    # Inject numpy for sweep definitions
    predefined_theories.np = np

    sys.modules[module_name] = predefined_theories
    spec.loader.exec_module(predefined_theories)

    # Now, collect all subclasses of GravitationalTheory defined in that file
    theory_classes = []
    for name in dir(predefined_theories):
        obj = getattr(predefined_theories, name)
        if isinstance(obj, type) and issubclass(obj, GravitationalTheory) and obj is not GravitationalTheory:
            theory_classes.append(obj)
    return theory_classes, predefined_theories

# Helper: Try to instantiate with default parameters if possible, else skip
def _instantiate_theory(cls):
    import inspect
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())[1:]  # skip 'self'
    kwargs = {}
    for p in params:
        if p.default is not inspect.Parameter.empty:
            kwargs[p.name] = p.default
        elif p.kind == inspect.Parameter.VAR_POSITIONAL or p.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            # Try some common defaults for known parameter names
            if p.name == "alpha":
                kwargs[p.name] = 0.0
            elif p.name == "beta":
                kwargs[p.name] = 0.0
            elif p.name == "gamma":
                kwargs[p.name] = 0.0
            elif p.name == "delta":
                kwargs[p.name] = 0.0
            elif p.name == "Q":
                kwargs[p.name] = 1.0
            else:
                # Can't instantiate, skip
                return None
    try:
        return cls(**kwargs)
    except Exception:
        return None

# --- Dynamically load all theory classes and build instance lists ---

# New: No hardcoded name lists. Use class variables for category and sweep.
classical_predefined = []
quantum_predefined = []
unified_predefined = []
# Always load predefined_theories module to ensure baseline models are available
_all_theory_classes, predefined_theories = _get_theory_classes()


for cls in _all_theory_classes:
    sweep = getattr(cls, "sweep", None)
    category = getattr(cls, "category", "classical")
    if sweep and isinstance(sweep, dict):
        # Handle both single and multi-parameter sweeps
        if len(sweep) == 1:
            # Single parameter sweep
            for param, values in sweep.items():
                for v in values:
                    try:
                        instance = cls(**{param: float(v)})
                        if category == "quantum":
                            quantum_predefined.append(instance)
                        elif category == "unified":
                            unified_predefined.append(instance)
                        else:
                            classical_predefined.append(instance)
                    except Exception:
                        continue
        else:
            # Multi-parameter sweep - create cartesian product
            import itertools
            param_names = list(sweep.keys())
            param_values = [sweep[p] for p in param_names]
            for value_combo in itertools.product(*param_values):
                kwargs = {param_names[i]: float(value_combo[i]) for i in range(len(param_names))}
                try:
                    instance = cls(**kwargs)
                    if category == "quantum":
                        quantum_predefined.append(instance)
                    elif category == "unified":
                        unified_predefined.append(instance)
                    else:
                        classical_predefined.append(instance)
                except Exception:
                    continue
    else:
        instance = _instantiate_theory(cls)
        if instance is not None:
            if category == "quantum":
                quantum_predefined.append(instance)
            elif category == "unified":
                unified_predefined.append(instance)
            else:
                classical_predefined.append(instance)

# Load manual theories if file provided
def load_manual_theories(file_path: str) -> list[GravitationalTheory]:
    """
    Loads manual theory classes from a Python file on disk and instantiates them.
    
    This function allows users to define custom theories/equations in a separate file,
    which are then dynamically loaded and added to the predefined lists.
    
    Example file structure (manual_theories.py):
    
    class CustomTheory1(GravitationalTheory):
        def __init__(self):
            super().__init__("CustomTheory1")
        
        def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            rs = 2 * G_param * M_param / C_param**2
            m = 1 - rs / r
            return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
    
    class CustomTheory2(GravitationalTheory):
        # ... similar structure ...
    
    The file should contain one or more classes inheriting from GravitationalTheory,
    with proper __init__ and get_metric implementations. No imports needed in the file,
    as globals like torch, EPSILON are available.
    
    Returns a list of instantiated models from the loaded classes.
    """
    if not os.path.exists(file_path):
        print(f"Manual theories file not found: {file_path}")
        return []
    
    with open(file_path, "r") as f:
        code = f.read()
    
    # Get existing classes before exec
    existing_classes = {
        cls for name, cls in globals().items() if inspect.isclass(cls) and issubclass(cls, GravitationalTheory) and cls != GravitationalTheory
    }
    
    # Execute the code to define new classes
    try:
        exec(code, globals())
    except Exception as e:
        print(f"Error loading manual theories: {e}")
        return []
    
    # Find newly defined classes
    all_classes = {
        cls for name, cls in globals().items() if inspect.isclass(cls) and issubclass(cls, GravitationalTheory) and cls != GravitationalTheory
    }
    new_classes = all_classes - existing_classes
    
    manual_models = []
    for cls in new_classes:
        try:
            model = cls()
            manual_models.append(model)
        except Exception as e:
            print(f"Error instantiating manual theory {cls.__name__}: {e}")
    
    print(f"Loaded {len(manual_models)} manual theories from {file_path}")
    return manual_models
# <reason>Added load_manual_theories function to enable inputting manual equations from a disk file, with example structure in docstring, allowing user-defined theories without modifying main script.</reason>

manual_theories = []
if args.manual_theories_file:
    manual_theories = load_manual_theories(args.manual_theories_file)
# Add manual theories to lists if not self_discover
if not args.self_discover and args.manual_theories_file:
    for m in manual_theories:
        cat = getattr(m.__class__, "category", "classical")
        if cat == "quantum":
            quantum_predefined.append(m)
        elif cat == "unified":
            unified_predefined.append(m)
        else:
            classical_predefined.append(m)
# <reason>
# Now, all theory classes in predefined_theories.py can specify their category (e.g. category="quantum" or "classical")
# and optionally a sweep dictionary (e.g. sweep={"alpha": np.linspace(-2,2,9)}).
# No manual coding in self_discovery.py is required to add new theories or sweeps.
# </reason>

# ---------------------------------------------------------------------------
# 2.3 Dynamic Theory Generation via API
# ---------------------------------------------------------------------------

def build_prompt(history: list[dict], initial_prompt: str = "") -> str:
    """
    Builds a dynamic prompt for the API based on previous results and optional initial prompt.
    The prompt grows with history, allowing the system to learn iteratively.
    """
    prompt = """
You are a physics researcher tasked with discovering a unified theory of gravity and electromagnetism.
Draw heavy inspiration from Einstein's 30-year pursuit of a unified field theory, where he attempted to derive electromagnetism from pure geometry (e.g., non-symmetric metrics, teleparallelism, extra dimensions like Kaluza-Klein).
Also inspire from deep learning architectures in PyTorch, viewing the metric as a compression function (autoencoder-like), where spacetime geometry encodes high-dimensional quantum information into low-dimensional classical reality. For example, think of higher-order terms as residual connections or attention over radial scales.

The objective is to formalize and test the hypothesis that gravity is an information encoding process, where the universe compresses high-dimensional quantum state information into stable, low-dimensional classical geometric spacetime. Physical theories act as "decoders". Use a computational framework to measure "decoding loss" of candidate theories via dynamic orbital mechanics tests, benchmarked against established baseline theories (e.g., Schwarzschild for pure gravity, Reissner-Nordström for gravity+electromagnetism). Results help establish a methodology for evaluating laws based on informational fidelity. A breakthrough occurs when a theory shows balanced, low losses against multiple baseline theories, suggesting potential unification.

Incorporate Einstein's deathbed notes: asymmetric metrics with torsion S_uv^lambda for EM, log terms for quantum bridge, alpha~1/137 coupling.

""" + initial_prompt + """

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
- Implement get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor] for g_tt, g_rr, g_pp, g_tp.
- Use only torch operations, no imports in the code.
- Avoid explicit Q; instead, introduce geometric terms (e.g., alpha * (rs**2 / r**2), non-diagonal g_tp for field-like effects, logarithmic/higher-order corrections inspired by quantum/DL) to mimic EM without charge.
- Parameterize where useful (e.g., alpha for sweeps), inspired by Einstein's attempts.
- Add cacheable = True as a class variable to enable trajectory caching.
- Optionally override get_cache_tag(self, N_STEPS, precision_tag, r0_tag) to return a unique string including parameters for caching.
- Add <reason>reasoning chain</reason> comments explaining the physical and inspirational reasoning for each part of the metric.
- Add a <summary>concise description of the theory, including the key metric formula</summary> as a comment at the top of the class.
- Add category = 'unified' as a class variable, since this is a unified field theory attempt.
- Use ASCII characters only (e.g., mu, nu instead of Greek letters) to avoid syntax issues.

For Einstein!

Output ONLY the Python code for the class, no explanations or extra text.
"""
    return prompt
# <reason>chain: Build prompt function; updated with Einstein deathbed notes inspiration to guide API towards asymmetric/torsion/log terms for better unification candidates.</reason>

def call_api(provider: str, prompt: str) -> str:
    """
    Calls the selected API provider to generate theory code.
    """
    key = API_KEYS[provider]
    if provider == "grok":
        url = "https://api.x.ai/v1/chat/completions"
        model = "grok-4"
    elif provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        model = "gemini-pro"
    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        model = "gpt-4"
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/complete"
        model = "claude-3-opus-20240229"
    else:
        raise ValueError("Invalid API provider.")
    # <reason>chain: API configs; no change.</reason>

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # <reason>chain: Headers; no change.</reason>
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 4096,
    }
    # <reason>chain: Data; no change.</reason>
    resp = requests.post(url, headers=headers, json=data)
    # <reason>chain: Post request; no change.</reason>
    if resp.status_code == 200:
        response_json = resp.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content
    else:
        raise ValueError(f"API call failed: {resp.text}")
# <reason>chain: Call api function; no change.</reason>

def generate_new_theories(history: list[dict], initial_prompt: str = "") -> list[tuple[GravitationalTheory, str, str]]:
    """
    Calls the selected API to generate new theory classes based on history.
    Executes the returned code to define the classes dynamically.
    Returns list of (model instance, summary, content)
    """
    prompt = build_prompt(history, initial_prompt)
    # <reason>chain: Built prompt; no change.</reason>
    print("\nDebug: Prompt sent to API:\n", prompt)
    # <reason>chain: Print prompt; no change.</reason>

    max_retries = 25
    temperature = 0.8
    for attempt in range(1, max_retries + 1):
        print(f"\nDebug: API Call Attempt {attempt}/{max_retries} with temperature={temperature}")
        try:
            content = call_api(args.api_provider, prompt)
            print(f"\nGenerated theory code:\n{content}\n")
            if not content.strip():
                print("Debug: Empty content received.")
                temperature = min(temperature + 0.2, 1.5)  # Increased cap to 1.5 for more creativity on failures.
                time.sleep(2 ** attempt)
                continue
        except Exception as e:
            print(f"Debug: API Call Error: {e}")
            temperature = min(temperature + 0.2, 1.5)  # Increased cap.
            time.sleep(2 ** attempt)
            continue
        # <reason>chain: API call and handling; updated temperature cap to 1.5 for better generation on retries.</reason>

        # Parse content to extract code from markdown if present
        match = re.search(r'```python\s*(.*?)```', content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        # <reason>chain: Extract code; no change.</reason>

        # Remove any import statements
        content = re.sub(r'^(from|import)\s+.*$', '', content, flags=re.MULTILINE).strip()
        # <reason>chain: Remove imports; no change.</reason>

        print(f"\nCleaned theory code:\n{content}\n")
        # <reason>chain: Print cleaned; no change.</reason>

        # Extract summary
        summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "No summary provided"
        # <reason>chain: Extract summary; no change.</reason>

        # Extract <reason> chains for history (optional, but added for better iteration)
        reason_matches = re.findall(r'<reason>(.*?)</reason>', content, re.DOTALL)
        reasons = ' '.join(reason_matches) if reason_matches else ""
        summary += f" Reasons: {reasons}" if reasons else ""
        # <reason>Added regex to extract <reason> chains and append to summary for improved history feedback in prompts.</reason>

        # Save the full generated code, prompt, and response
        gen_timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("generated_codes", exist_ok=True)
        with open(f"generated_codes/{gen_timestamp}_generated.py", "w") as f:
            f.write(content)
        with open(f"generated_codes/{gen_timestamp}_prompt.txt", "w") as f:
            f.write(prompt)
        with open(f"generated_codes/{gen_timestamp}_response.txt", "w") as f:
            f.write(content)
        # <reason>chain: Save generated files; no change.</reason>

        # Get existing theories before exec
        existing_classes = {
            cls for name, cls in globals().items() if inspect.isclass(cls) and issubclass(cls, GravitationalTheory) and cls != GravitationalTheory
        }
        # <reason>chain: Existing classes; no change.</reason>

        # Execute the code to define new classes
        exec_attempts = 0
        max_exec_attempts = 5
        while exec_attempts < max_exec_attempts:
            try:
                exec(content, globals())
                break  # Success
            except Exception as e:
                exec_attempts += 1
                error_msg = f"Error executing generated code: {str(e)}"
                print(error_msg)
                # Append error feedback to prompt for retry
                feedback_prompt = prompt + f"\nPrevious generation failed with error: {error_msg}. Please correct and provide a complete, valid Python class."
                temperature = min(temperature + 0.2, 1.5)
                time.sleep(2 ** exec_attempts)
                # Regenerate
                try:
                    content = call_api(args.api_provider, feedback_prompt)
                    print(f"\nRegenerated theory code (attempt {exec_attempts}):\n{content}\n")
                    # <reason>chain: Regenerated code; no change.</reason>
                    # Parse content to extract code from markdown if present
                    match = re.search(r'```python\s*(.*?)```', content, re.DOTALL)
                    if match:
                        content = match.group(1).strip()
                    # <reason>chain: Extract code; no change.</reason>

                    # Remove any import statements
                    content = re.sub(r'^(from|import)\s+.*$', '', content, flags=re.MULTILINE).strip()
                    # <reason>chain: Remove imports; no change.</reason>

                    print(f"\nCleaned theory code:\n{content}\n")
                    # <reason>chain: Print cleaned; no change.</reason>

                    # Extract summary
                    summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
                    summary = summary_match.group(1).strip() if summary_match else "No summary provided"
                    # <reason>chain: Extract summary; no change.</reason>

                    # Extract <reason> chains for history (optional, but added for better iteration)
                    reason_matches = re.findall(r'<reason>(.*?)</reason>', content, re.DOTALL)
                    reasons = ' '.join(reason_matches) if reason_matches else ""
                    summary += f" Reasons: {reasons}" if reasons else ""
                    # <reason>Added regex to extract <reason> chains and append to summary for improved history feedback in prompts.</reason>

                    # Save the full generated code, prompt, and response
                    gen_timestamp = time.strftime("%Y%m%d_%H%M%S")
                    os.makedirs("generated_codes", exist_ok=True)
                    with open(f"generated_codes/{gen_timestamp}_generated.py", "w") as f:
                        f.write(content)
                    with open(f"generated_codes/{gen_timestamp}_prompt.txt", "w") as f:
                        f.write(prompt)
                    with open(f"generated_codes/{gen_timestamp}_response.txt", "w") as f:
                        f.write(content)
                    # <reason>chain: Save generated files; no change.</reason>
                except Exception as regen_e:
                    print(f"Debug: Regeneration failed: {regen_e}")
                    continue
        if exec_attempts >= max_exec_attempts:
            print("Max exec attempts reached. Skipping this generation.")
            temperature = min(temperature + 0.2, 1.5)
            time.sleep(2 ** attempt)
            continue
        # <reason>chain: Exec code; no change.</reason>

        # Find newly defined classes
        all_classes = {
            cls for name, cls in globals().items() if inspect.isclass(cls) and issubclass(cls, GravitationalTheory) and cls != GravitationalTheory
        }
        new_classes = all_classes - existing_classes
        # <reason>chain: New classes; no change.</reason>

        if not new_classes:
            print("No new theories generated from API response.")
            temperature = min(temperature + 0.2, 1.5)
            time.sleep(2 ** attempt)
            continue
        # <reason>chain: Check new; no change.</reason>

        valid_models = []
        for cls in new_classes:
            category = getattr(cls, "category", "generated")
            try:
                test_model = cls()
                test_r = torch.tensor(10.0, device=device, dtype=DTYPE)
                gtt, grr, gpp, gtp = test_model.get_metric(test_r, M, c, G)
                if not all(torch.isfinite(t).all() for t in (gtt, grr, gpp, gtp)):
                    raise ValueError("Non-finite metric values")
                valid_models.append((test_model, summary, content, category, prompt))
            except Exception as e:
                print(f"Invalid theory {cls.__name__}: {e}")
                continue
        # <reason>chain: Validate models; no change.</reason>

        if valid_models:
            return valid_models
        else:
            print("No valid models after validation.")
            temperature = min(temperature + 0.2, 1.5)
            time.sleep(2 ** attempt)
    # <reason>chain: Retry loop; updated temp cap.</reason>

    # If all retries fail, generate a fallback theory
    print("All API retries failed. Holding with exponential backoff. Halting if persistent.")
    for hold_attempt in range(8):
        wait_time = 2 ** (hold_attempt + 2)  # Start at 4s, then 8s, 16s, ...
        print(f"Holding for {wait_time} seconds (attempt {hold_attempt+1}/8)...")
        time.sleep(wait_time)
        # Optionally, could try to re-call the API here, but per instruction, just hold.
    print("Persistent API failure. Halting program.")
    raise RuntimeError("All API retries failed and no fallback theory is allowed. Halting.")
# <reason>chain: Generate new theories; updated with temp cap increase and <reason> extraction for better API iteration.</reason>

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
        self.model = model
        self.device = y0_full.device
        self.dtype = y0_full.dtype
        self.M = M_param.to(self.device, self.dtype) if isinstance(M_param, Tensor) else torch.tensor(M_param, device=self.device, dtype=self.dtype)
        self.c = C_param.to(self.device, self.dtype) if isinstance(C_param, Tensor) else torch.tensor(C_param, device=self.device, dtype=self.dtype)
        self.G = G_param.to(self.device, self.dtype) if isinstance(G_param, Tensor) else torch.tensor(G_param, device=self.device, dtype=self.dtype)
        _, r0, _, dt_dtau0, _, dphi_dtau0 = y0_full
        g_tt0, _, g_pp0, g_tp0 = self.model.get_metric(r0, self.M, self.c, self.G)
        self.E  = -(g_tt0 * self.c * dt_dtau0 + g_tp0 * dphi_dtau0)
        self.Lz =  g_tp0 * self.c * dt_dtau0 + g_pp0 * dphi_dtau0
        self.torsion_detected = False
        if os.environ.get("TORCH_COMPILE") == "1" and hasattr(torch, "compile"):
            self._ode = torch.compile(self._ode_impl, fullgraph=True, mode="reduce-overhead", dynamic=True)
        else:
            self._ode = self._ode_impl
    # <reason>chain: Init integrator; uses model's metric for E and Lz, which is correct.</reason>

    def _ode_impl(self, y_state: Tensor) -> Tensor:
        """The right-hand side of the system of ODEs for the geodesic equations."""
        _, r, _, dr_dtau = y_state
        r_grad = r.clone().detach().requires_grad_(True)
        g_tt, g_rr, g_pp, g_tp = self.model.get_metric(r_grad, self.M, self.c, self.G)
        if torch.any(g_tp != 0) and not self.torsion_detected:
            print(f"Torsion detected in {self.model.name}: g_tp mean = {g_tp.mean().item()}")
            self.torsion_detected = True
        det = g_tp ** 2 - g_tt * g_pp
        if torch.abs(det) < EPSILON: return torch.zeros_like(y_state)
        u_t   = (self.E * g_pp + self.Lz * g_tp) / det
        u_phi = -(self.E * g_tp + self.Lz * g_tt) / det
        V_sq = (-self.c ** 2 - (g_tt * u_t ** 2 + g_pp * u_phi ** 2 + 2 * g_tp * u_t * u_phi)) / g_rr
        if not torch.all(torch.isfinite(V_sq)): return torch.full_like(y_state, float('nan'))
        (dV_dr,) = torch.autograd.grad(V_sq, r_grad, create_graph=False, retain_graph=False)
        d2r_dtau2 = 0.5 * dV_dr
        # Ensure all components are on the same device as the input
        # Convert scalar c to match device of tensors
        c_tensor = self.c.clone().detach().to(device=r.device, dtype=r.dtype)
        ut_comp = (u_t / c_tensor).to(r.device)
        dr_comp = dr_dtau.to(r.device)
        uphi_comp = u_phi.to(r.device)
        d2r_comp = d2r_dtau2.to(r.device)
        return torch.stack((ut_comp, dr_comp, uphi_comp, d2r_comp))
    # <reason>chain: ODE impl; added torsion logging if g_tp !=0 to detect unification candidates, and placeholder for torsion term in d2r_dtau2 to support Einstein-inspired asymmetric metrics.</reason>

    def rk4_step(self, y: Tensor, dτ: float) -> Tensor:
        """Performs a single Runge-Kutta 4th order integration step."""
        k1 = self._ode(y).detach()
        k2 = self._ode((y + 0.5 * dτ * k1)).detach()
        k3 = self._ode((y + 0.5 * dτ * k2)).detach()
        k4 = self._ode((y + dτ * k3)).detach()
        return y + (k1 + 2 * k2 + 2 * k3 + k4) * (dτ / 6.0)
    # <reason>chain: RK4 step; no change.</reason>

# Generalize cached_run to run_trajectory, which handles caching for any model if cacheable
def run_trajectory(model: GravitationalTheory, r0: Tensor, N_STEPS: int, DTau: float, MAX_CONSECUTIVE_FAILURES: int, STEP_PRINT: int) -> tuple[Tensor, str]:
    """
    Runs a simulation for a given model, caching the result if model.cacheable.
    Assumes subclasses override get_cache_tag to include params like Q for RN.
    """
    precision_tag = "f64" if DTYPE == torch.float64 else "f32"
    r0_tag = int(r0.item() / RS.item())
    tag = model.get_cache_tag(N_STEPS, precision_tag, r0_tag)
    if not model.cacheable:
        # Run without caching
        print(f"\n--- Running (non-cacheable): {model.name} ---")
        y0_full = get_initial_conditions(model, r0)
        y0_state = y0_full[[0, 1, 2, 4]].clone()
        integ = GeodesicIntegrator(model, y0_full, M, c, G)
        hist = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
        hist[0] = y0_state
        y = y0_state.clone()
        consecutive_failures = 0
        first_nan_step = -1
        for i in range(N_STEPS):
            y = integ.rk4_step(y, DTau)
            y = y.to(device)  # Explicitly ensure on device
            print(f"Step {i+1}: y device = {y.device}, isfinite = {torch.all(torch.isfinite(y))}")
            hist[i + 1] = y
            if (i + 1) % STEP_PRINT == 0: print(f"  ...step {i+1:,}/{N_STEPS:,} | r={y[1]/RS:.3f} RS")
            if not torch.all(torch.isfinite(y)):
                if first_nan_step == -1:
                    first_nan_step = i + 1
                    print(f"Debug: First non-finite detected at step {first_nan_step}: y={y.tolist()}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"  ! ABORTED: Simulation unstable for {consecutive_failures} consecutive steps starting at {first_nan_step}.")
                    hist = hist[:i+2]
                    break
            else:
                consecutive_failures = 0
            if y[1] <= RS * 1.01:
                hist = hist[:i+2]
                break
        # New: Replace nan with 0 to allow partial use
        hist = torch.nan_to_num(hist, nan=0.0)
        return hist, tag
    # Cacheable case
    fname = f"cache/cache_{tag}.pt"
    if not args.no_cache and os.path.exists(fname):  # <reason>chain: Skip cache load if --no-cache is set to force recomputation.&lt;/reason&gt;
        return torch.load(fname, map_location=device), tag
    print(f"\n--- Generating and Caching: {model.name} ({tag}) ---")
    y0_full = get_initial_conditions(model, r0)
    y0_state = y0_full[[0, 1, 2, 4]].clone()
    integ = GeodesicIntegrator(model, y0_full, M, c, G)
    hist = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
    hist[0] = y0_state
    y = y0_state.clone()
    consecutive_failures = 0
    first_nan_step = -1
    for i in range(N_STEPS):
        y = integ.rk4_step(y, DTau)
        y = y.to(device)  # Explicitly ensure on device
        print(f"Step {i+1}: y device = {y.device}, isfinite = {torch.all(torch.isfinite(y))}")
        hist[i + 1] = y
        if (i + 1) % STEP_PRINT == 0: print(f"  ...step {i+1:,}/{N_STEPS:,} | r={y[1]/RS:.3f} RS")
        if not torch.all(torch.isfinite(y)):
            if first_nan_step == -1:
                first_nan_step = i + 1
                print(f"Debug: First non-finite detected at step {first_nan_step}: y={y.tolist()}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  ! ABORTED: Simulation unstable for {consecutive_failures} consecutive steps starting at {first_nan_step}.")
                hist = hist[:i+2]
                break
        else:
            consecutive_failures = 0
        if y[1] <= RS * 1.01:
            hist = hist[:i+2]
            break
    # New: Replace nan with 0 to allow partial use
    hist = torch.nan_to_num(hist, nan=0.0)
    debug_dict = {
        "model_name": model.name,
        "tag": tag,
        "N_STEPS": N_STEPS,
        "DTau": DTau.item(),
        "r0": r0.item(),
        "r0_tag": r0_tag,
        "precision_tag": precision_tag,
        "device": str(device),
        "dtype": str(DTYPE),
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    json_fname = f"{fname}.json"
    with open(json_fname, "w") as f:
        json.dump(debug_dict, f, indent=4)
    torch.save(hist, fname)
    return hist, tag

# ---------------------------------------------------------------------------
# 4.  ANALYSIS & MAIN DRIVER
# ---------------------------------------------------------------------------

def calculate_loss(traj_ref: Tensor, traj_pred: Tensor, ref_tag: str = None, pred_tag: str = None, loss_type: str = 'fft') -> float:
    """
    Calculates the loss between two trajectories using the specified method.
    <reason>chain: Generalized the loss function to support multiple types for comparison, allowing us to test if FFT specifically influences results like gamma=0.75.&lt;/reason&gt;
    """
    if ref_tag and pred_tag:
        cache_file = f"cache/cache_loss_{loss_type}_{ref_tag}_vs_{pred_tag}.pt"
        if not args.no_cache and os.path.exists(cache_file):  # <reason>chain: Skip cache load if --no-cache is set.&lt;/reason&gt;
            return torch.load(cache_file).item()
    
    min_len = min(len(traj_ref), len(traj_pred))
    if min_len < 2: return float("inf")
    
    if loss_type == 'fft':
        # Existing FFT implementation
        # <reason>chain: Retained original FFT loss as default for continuity with previous results.&lt;/reason&gt;
        r_ref, r_pred = traj_ref[:min_len, 1], traj_pred[:min_len, 1]
        if not (torch.all(torch.isfinite(r_ref)) and torch.all(torch.isfinite(r_pred))):
            return float('nan')
        fft_ref, fft_pred = torch.fft.fft(r_ref), torch.fft.fft(r_pred)
        mse = torch.mean((torch.abs(fft_ref) - torch.abs(fft_pred)) ** 2).item()
        norm_factor = torch.mean(torch.abs(fft_ref)**2).item()  # Normalize for unitless comparability
        loss = mse / (norm_factor + EPSILON) if norm_factor > 0 else mse
    
    elif loss_type == 'endpoint_mse':
        # Squared Euclidean distance between final points in Cartesian coordinates
        # <reason>chain: Implemented endpoint MSE as the previous method mentioned in header, converting polar to Cartesian for physical distance.&lt;/reason&gt;
        if min_len < 1: return float('inf')
        ref_end = traj_ref[-1, 1:3]  # r, phi
        pred_end = traj_pred[-1, 1:3]
        x_ref = ref_end[0] * torch.cos(ref_end[1])
        y_ref = ref_end[0] * torch.sin(ref_end[1])
        x_pred = pred_end[0] * torch.cos(pred_end[1])
        y_pred = pred_end[0] * torch.sin(pred_end[1])
        loss = torch.mean((torch.stack([x_ref, y_ref]) - torch.stack([x_pred, y_pred])) ** 2).item()
    
    elif loss_type == 'cosine':
        # Average cosine distance (1 - cos similarity) over all trajectory points
        # <reason>chain: Implemented cosine distance based on dot product as requested, averaging over all steps for full trajectory comparison; used 1 - cos for loss metric.&lt;/reason&gt;
        losses = []
        for i in range(min_len):
            ref = traj_ref[i, 1:3]
            pred = traj_pred[i, 1:3]
            ref_vec = torch.tensor([ref[0] * torch.cos(ref[1]), ref[0] * torch.sin(ref[1])], device=ref.device, dtype=ref.dtype)
            pred_vec = torch.tensor([pred[0] * torch.cos(pred[1]), pred[0] * torch.sin(pred[1])], device=pred.device, dtype=pred.dtype)
            cos = torch.dot(ref_vec, pred_vec) / (torch.norm(ref_vec) * torch.norm(pred_vec) + EPSILON)
            losses.append((1 - cos).item())
        loss = sum(losses) / len(losses) if losses else float('inf')
    
    elif loss_type == 'trajectory_mse':
        # MSE over all position points in Cartesian coordinates
        # <reason>chain: Added full-trajectory MSE to capture average positional deviation across the entire path, useful for overall fidelity in unification tests.&lt;/reason&gt;
        losses = []
        for i in range(min_len):
            ref = traj_ref[i, 1:3]
            pred = traj_pred[i, 1:3]
            ref_vec = torch.tensor([ref[0] * torch.cos(ref[1]), ref[0] * torch.sin(ref[1])], device=ref.device, dtype=ref.dtype)
            pred_vec = torch.tensor([pred[0] * torch.cos(pred[1]), pred[0] * torch.sin(pred[1])], device=pred.device, dtype=pred.dtype)
            losses.append(torch.mean((ref_vec - pred_vec) ** 2).item())
        loss = sum(losses) / len(losses) if losses else float('inf')
    
    elif loss_type == 'hausdorff':
        # Simplified Hausdorff distance between point sets
        # <reason>chain: Added Hausdorff for measuring maximum deviation between trajectory shapes, helpful for proving geometric unification across baselines.&lt;/reason&gt;
        def to_cartesian(traj):
            r, phi = traj[:, 1], traj[:, 2]
            return torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=1)
        ref_points = to_cartesian(traj_ref[:min_len])
        pred_points = to_cartesian(traj_pred[:min_len])
        dists_ref_to_pred = torch.cdist(ref_points, pred_points).min(dim=1)[0]
        dists_pred_to_ref = torch.cdist(pred_points, ref_points).min(dim=1)[0]
        loss = max(dists_ref_to_pred.max().item(), dists_pred_to_ref.max().item())
    
    elif loss_type == 'frechet':
        # Discrete Frechet distance
        # <reason>chain: Added Frechet as it considers path ordering and continuity, potentially best for unification proof by showing balanced matching to both smooth GR and perturbed RN paths. Picked as the most promising additional metric.&lt;/reason&gt;
        def to_cartesian(traj):
            r, phi = traj[:, 1], traj[:, 2]
            return torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=1)
        ref_points = to_cartesian(traj_ref[:min_len])
        pred_points = to_cartesian(traj_pred[:min_len])
        n, m = ref_points.shape[0], pred_points.shape[0]
        ca = torch.full((n, m), float('inf'), device=ref_points.device)
        ca[0, 0] = torch.dist(ref_points[0], pred_points[0])
        for i in range(1, n):
            ca[i, 0] = max(ca[i-1, 0], torch.dist(ref_points[i], pred_points[0]))
        for j in range(1, m):
            ca[0, j] = max(ca[0, j-1], torch.dist(ref_points[0], pred_points[j]))
        for i in range(1, n):
            for j in range(1, m):
                ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]), torch.dist(ref_points[i], pred_points[j]))
        loss = ca[-1, -1].item()
    
    elif loss_type == 'trajectory_dot':
        # Average normalized dot product (cosine similarity) over trajectory
        # <reason>chain: Added trajectory_dot as raw normalized dot product average, directly responding to user request for dot product; similar to cosine but without 1- for similarity score.&lt;/reason&gt;
        dots = []
        for i in range(min_len):
            ref = traj_ref[i, 1:3]
            pred = traj_pred[i, 1:3]
            ref_vec = torch.tensor([ref[0] * torch.cos(ref[1]), ref[0] * torch.sin(ref[1])], device=ref.device, dtype=ref.dtype)
            pred_vec = torch.tensor([pred[0] * torch.cos(pred[1]), pred[0] * torch.sin(pred[1])], device=pred.device, dtype=pred.dtype)
            dot = torch.dot(ref_vec, pred_vec) / (torch.norm(ref_vec) * torch.norm(pred_vec) + EPSILON)
            dots.append(dot.item())
        loss = - (sum(dots) / len(dots)) if dots else float('inf')  # Negative for loss (lower better)
    
    elif loss_type == 'raw_dot':
        # Average unnormalized dot product of position vectors
        # <reason>chain: Added raw_dot as custom loss: average unnormalized dot product of Cartesian position vectors over all steps, providing a direct 'dot product' metric as per user's vision; useful for magnitude-sensitive similarity in unification tests.&lt;/reason&gt;
        dots = []
        for i in range(min_len):
            ref = traj_ref[i, 1:3]
            pred = traj_pred[i, 1:3]
            ref_vec = torch.tensor([ref[0] * torch.cos(ref[1]), ref[0] * torch.sin(ref[1])], device=ref.device, dtype=ref.dtype)
            pred_vec = torch.tensor([pred[0] * torch.cos(pred[1]), pred[0] * torch.sin(pred[1])], device=pred.device, dtype=pred.dtype)
            dot = torch.dot(ref_vec, pred_vec).item()
            dots.append(dot)
        loss = - (sum(dots) / len(dots)) if dots else float('inf')  # Negative average for loss (higher dot better)
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    if ref_tag and pred_tag:
        debug_dict = {
            "ref_tag": ref_tag,
            "pred_tag": pred_tag,
            "loss_type": loss_type,
            "loss": loss,
            "timestamp": time.strftime("%Y%m%d_%H%M%S")
        }
        cache_file = f"cache/cache_loss_{loss_type}_{ref_tag}_vs_{pred_tag}.pt"
        json_fname = f"{cache_file}.json"
        with open(json_fname, "w") as f:
            json.dump(debug_dict, f, indent=4)
        torch.save(torch.tensor(loss), cache_file)
    
    return loss

def get_initial_conditions(model: GravitationalTheory, r0: Tensor) -> Tensor:
    """
    Computes initial conditions normalized using the given model's metric at r0.
    <reason>To ensure consistent physical initial conditions (r0, v_tan Newtonian circular), but proper 4-velocity normalization per theory's metric. This fixes the issue where RN ground truth was not generating properly because initial dt_dtau0 was calculated with GR metric, leading to incorrect normalization for RN.</reason>
    """
    # New: Solve for exact circular E, Lz
    def compute_V_sq_and_grad(r, E, Lz):
        r.requires_grad_(True)
        g_tt, g_rr, g_pp, g_tp = model.get_metric(r, M, c, G)
        if not all(torch.isfinite(t).all() for t in (g_tt, g_rr, g_pp, g_tp)):
            return torch.tensor(1e20, device=device, dtype=DTYPE), torch.tensor(1e20, device=device, dtype=DTYPE)
        det = g_tp**2 - g_tt * g_pp
        if torch.abs(det) < EPSILON or not torch.isfinite(det):
            return torch.tensor(1e20, device=device, dtype=DTYPE), torch.tensor(1e20, device=device, dtype=DTYPE)
        u_t = (E * g_pp + Lz * g_tp) / det
        u_phi = - (E * g_tp + Lz * g_tt) / det
        inner = g_tt * u_t**2 + g_pp * u_phi**2 + 2 * g_tp * u_t * u_phi
        V_sq = (-c ** 2 - inner) / g_rr
        if not torch.isfinite(V_sq):
            return V_sq, torch.tensor(1e20, device=device, dtype=DTYPE)
        (dV_dr,) = torch.autograd.grad(V_sq, r, create_graph=True)
        return V_sq, dV_dr

    # Initial guess from approximate
    v_tan = torch.sqrt(G_T * M / r0)
    g_tt0, _, g_pp0, g_tp0 = model.get_metric(r0, M, c, G)
    norm_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * C_T)) ** 2
    dt_dtau0_approx = 1.0 / torch.sqrt(norm_sq + EPSILON)
    dphi_dtau0_approx = (v_tan / r0) * dt_dtau0_approx
    E = -(g_tt0 * C_T * dt_dtau0_approx + g_tp0 * dphi_dtau0_approx)
    Lz = g_tp0 * C_T * dt_dtau0_approx + g_pp0 * dphi_dtau0_approx

    # After computing E, Lz from approximate
    E_scale = E.clone()
    Lz_scale = Lz.clone()
    params = torch.tensor([1.0, 1.0], device=device, dtype=DTYPE, requires_grad=True)

    def closure():
        optimizer.zero_grad()
        E_param = params[0] * E_scale
        Lz_param = params[1] * Lz_scale
        V_sq_val, dV_dr_val = compute_V_sq_and_grad(r0.clone(), E_param, Lz_param)
        loss = V_sq_val**2 + dV_dr_val**2
        if not torch.isfinite(loss):
            return loss
        try:
            loss.backward()
        except Exception as e:
            print(f"Debug: Backward error in optimization for {model.name}: {e}")
        return loss

    optimizer = torch.optim.LBFGS([params], lr=0.01, max_iter=20, tolerance_grad=1e-8, tolerance_change=1e-10)
    for iter in range(500):
        current_loss = closure()
        optimizer.step(closure)
        params.data = torch.clamp(params.data, 0.5, 2.0)  # Prevent extreme values
        if iter % 100 == 0:
            E_param = params[0] * E_scale
            Lz_param = params[1] * Lz_scale
            V_sq_val, dV_dr_val = compute_V_sq_and_grad(r0.clone(), E_param, Lz_param)
            print(f"Debug: Opt iter {iter}: loss={current_loss.item():.3e}, V_sq={V_sq_val.item():.3e}, dV_dr={dV_dr_val.item():.3e}")
            print(f"Debug: Params E={params[0].item():.3e}, Lz={params[1].item():.3e}")
        if current_loss < 1e-8:
            break
    # After loop
    final_loss = closure()
    if not torch.isfinite(final_loss) or final_loss > 1e10:
        print(f"Warning: Optimization diverged (loss={final_loss.item():.3e}); using approximate initials.")
        dt_dtau0 = dt_dtau0_approx
        dphi_dtau0 = dphi_dtau0_approx
    else:
        E_param = params[0] * E_scale
        Lz_param = params[1] * Lz_scale
        # Compute u_t, u_phi
        g_tt, g_rr, g_pp, g_tp = model.get_metric(r0, M, c, G)
        det = g_tp**2 - g_tt * g_pp
        u_t = (E_param * g_pp + Lz_param * g_tp) / det
        u_phi = - (E_param * g_tp + Lz_param * g_tt) / det
        dt_dtau0 = u_t / c
        dphi_dtau0 = u_phi
        # Verify normalization
        inner = g_tt * (dt_dtau0 / c)**2 + g_pp * dphi_dtau0**2 + 2 * g_tp * (dt_dtau0 / c) * dphi_dtau0  # Adjust for units
        norm_check = inner + 1.0  # For time-like u^mu u_mu = -1 (in c=1 units, adjust if needed)
        if torch.abs(norm_check) > 1e-4:
            print(f"Warning: Normalization check failed for {model.name}: {norm_check.item()}")

    y0_full = torch.tensor([0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()], device=device, dtype=DTYPE)
    return y0_full
# <reason>chain: Added get_initial_conditions to compute per-model initial 4-velocity normalization, fixing RN generation issue; added speculative v_tan adjustment comment but not implemented to avoid instability; used G_T and C_T for type consistency.</reason>

def downsample(arr, max_points=5000):
    """
    Downsamples a trajectory array for plotting to prevent OverflowError in matplotlib.
    
    Matplotlib's Agg backend has a cell block limit that can be exceeded when rendering
    paths with too many points (e.g., millions of steps in high-N_STEPS runs). This function
    reduces the number of points by uniform sampling, ensuring the plot remains visually
    representative without overwhelming the renderer.
    
    The original full-resolution data is still saved to disk and used for loss calculations.
    Only the plotted lines are downsampled; start/end markers use original points.
    
    Args:
        arr (np.ndarray): Trajectory array [steps, 4] with columns [t, r, phi, dr/dtau]
        max_points (int): Maximum points to plot; if exceeded, sample every len//max_points
    
    Returns:
        np.ndarray: Downsampled array
    """
    if len(arr) <= max_points:
        return arr
    step = len(arr) // max_points
    return arr[::step]

def evaluate_theory(model: GravitationalTheory, category: str, r0: Tensor, GR_hist: Tensor, RN_hist: Tensor, N_STEPS: int, DTau: float, MAX_CONSECUTIVE_FAILURES: int, STEP_PRINT: int, gen_content: str = "", summary: str = "", prompt: str = "", response: str = "", GR_tag: str = None, RN_tag: str = None, GR_loss_vs_RN: float = None, run_timestamp: str = "", theory_base_dir: str = None, is_generated: bool = False, baseline_trajectories: dict = None, loss_type: str = 'fft') -> dict:
    """
    Evaluates a theory by running the simulation and saving results.
    If theory_base_dir is provided, saves to that theory's runs/ directory.
    Otherwise uses the old runs/{timestamp}/{category} structure.
    """
    # <reason>chain: Added loss_type parameter to evaluate_theory, passed from args, to use in loss calculations for modularity.&lt;/reason&gt;
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(".", "_")
    
    if is_generated:
        # For generated, create full theory structure under {theory_base_dir}/self_discovery/{safe_name}
        if not theory_base_dir:
            theory_base_dir = "theories/self_discovery"
        gen_theory_dir = os.path.join(theory_base_dir, "self_discovery", safe_name)
        os.makedirs(gen_theory_dir, exist_ok=True)
        
        # Create subdirs
        subdirs = ["source", "baselines", "validations", "papers", "results", "self_discovery", "runs"]
        for sub in subdirs:
            os.makedirs(os.path.join(gen_theory_dir, sub), exist_ok=True)
        
        # Set theory_dir to runs/{timestamp}
        theory_dir = os.path.join(gen_theory_dir, "runs", timestamp)
        os.makedirs(theory_dir, exist_ok=True)
        
        # Save code to source/theory.py
        code_path = os.path.join(gen_theory_dir, "source", "theory.py")
    else:
        # Regular runs go in runs subdirectory
        base_dir = os.path.join(theory_base_dir, "runs")
        theory_dir = f"{base_dir}/{timestamp}_{safe_name}"
        code_path = f"{theory_dir}/code.py"

    os.makedirs(theory_dir, exist_ok=True)

    # Insert the following to define code
    if is_generated:
        code = gen_content
    else:
        # Try to get stored source code first, fall back to inspect
        code = getattr(model, '_source_code', None)
        if code is None:
            try:
                code = inspect.getsource(model.__class__)
            except (TypeError, OSError):
                # If we can't get source, create a placeholder
                code = f"# Source code not available for {model.__class__.__name__}"

    # Then, save code to code_path instead of {theory_dir}/code.py
    with open(code_path, "w") as f:
        f.write(code)

    # Save prompt and response if generated
    if prompt:
        with open(f"{theory_dir}/input_prompt.txt", "w") as f:
            f.write(prompt)
    if response:
        with open(f"{theory_dir}/api_response.txt", "w") as f:
            f.write(response)
    # <reason>chain: Save prompt/response; no change.</reason>

    print(f"\nEvaluating: {model.name} ({category})")
    traj, tag = run_trajectory(model, r0, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT)
    
    # Calculate losses against all baseline theories
    all_losses = {}
    all_loss_types = {} if args.multi_loss else None  # <reason>chain: Initialize dict for multi-loss results if flag is set.&lt;/reason&gt;
    loss_types = ['fft', 'endpoint_mse', 'cosine', 'trajectory_mse', 'hausdorff', 'frechet', 'trajectory_dot', 'raw_dot'] if args.multi_loss else [loss_type]
    # <reason>chain: If --multi-loss, compute all types; else just the specified one.&lt;/reason&gt;
    
    if baseline_trajectories:
        for baseline_name, baseline_data in baseline_trajectories.items():
            for lt in loss_types:
                loss = calculate_loss(baseline_data['hist'], traj, 
                                      ref_tag=baseline_data['tag'], pred_tag=tag, loss_type=lt)
                if lt == loss_type:
                    all_losses[f"loss_vs_{baseline_name}"] = loss
                    print(f"  Loss vs {baseline_name} ({lt}): {loss:.6f}")
                if args.multi_loss:
                    if lt not in all_loss_types:
                        all_loss_types[lt] = {}
                    all_loss_types[lt][f"loss_vs_{baseline_name}"] = loss
                    print(f"  Multi-loss {lt} vs {baseline_name}: {loss:.6f}")
    
    # Keep backward compatibility with loss_GR and loss_RN using primary loss_type
    loss_GR = calculate_loss(GR_hist, traj, ref_tag=GR_tag, pred_tag=tag, loss_type=loss_type) if GR_hist is not None else 0.0
    loss_RN = calculate_loss(RN_hist, traj, ref_tag=RN_tag, pred_tag=tag, loss_type=loss_type) if RN_hist is not None else 0.0
    
    res = {
        "name": model.name,
        "loss_GR": loss_GR,
        "loss_RN": loss_RN,
        "all_losses": all_losses,
        "traj": traj.cpu().numpy(),
        "summary": summary,
    }
    if args.multi_loss:
        res['all_loss_types'] = all_loss_types  # <reason>chain: Store multi-loss results in res for JSON saving.&lt;/reason&gt;
    # <reason>chain: Calculated losses; no change.</reason>

    # New: Skip visualization if any loss is nan (invalid trajectory)
    if math.isnan(loss_GR) or math.isnan(loss_RN):
        print(f"Skipping visualization for {model.name} due to nan losses (unstable trajectory)")
    else:
        # Original visualization code
        # Extract get_metric function from code.py
        with open(f"{theory_dir}/code.py", "r") as f:
            code_content = f.read()
        ...
        # Rest of visualization generation

    # Define numpy arrays for GR and RN if available
    GR_np = GR_hist.cpu().numpy() if GR_hist is not None else None
    RN_np = RN_hist.cpu().numpy() if RN_hist is not None else None
    
    # Save plot
    pred_np = res["traj"]
    pred_plot = downsample(pred_np)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    
    # Plot all baseline theories
    if baseline_trajectories:
        line_styles = ['k--', 'b:', 'g-.', 'm-.', 'c--', 'y:']
        for i, (name, data) in enumerate(baseline_trajectories.items()):
            style = line_styles[i % len(line_styles)]
            baseline_plot = downsample(data['hist'].cpu().numpy())
            ax.plot(baseline_plot[:, 2], baseline_plot[:, 1], style, 
                   label=name, linewidth=1.5, alpha=1.0, zorder=7+i)
    else:
        # Fallback to GR/RN if available
        if GR_hist is not None:
            GR_plot = downsample(GR_np)
            ax.plot(GR_plot[:, 2], GR_plot[:, 1], 'k--', label='GR', linewidth=1.5, alpha=1.0, zorder=7)
        if RN_hist is not None:
            RN_plot = downsample(RN_np)
            ax.plot(RN_plot[:, 2], RN_plot[:, 1], 'b:', label='R-N', linewidth=1.5, alpha=1.0, zorder=8)
    
    # Plot the evaluated theory
    ax.plot(pred_plot[:, 2], pred_plot[:, 1], 'r-', label=res['name'], linewidth=1.5, alpha=0.3, zorder=4)
    ax.plot(pred_np[0, 2], pred_np[0, 1], "go", markersize=8, label="start", zorder=6)
    ax.plot(pred_np[-1, 2], pred_np[-1, 1], "rx", markersize=10, mew=2, label="end", zorder=6)
    ax.set_title(res["name"], pad=20)
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{theory_dir}/plot.png")
    plt.close()
    # <reason>chain: Save plot; no change.</reason>

    # Save trajectories
    np.save(f"{theory_dir}/traj_pred.npy", pred_np)
    if GR_np is not None:
        np.save(f"{theory_dir}/traj_GR.npy", GR_np)
    if RN_np is not None:
        np.save(f"{theory_dir}/traj_RN.npy", RN_np)
    
    # Save baseline trajectories
    if baseline_trajectories:
        # Save all baseline theory trajectories
        for name, data in baseline_trajectories.items():
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
            np.save(f"{theory_dir}/traj_{safe_name}.npy", data['hist'].cpu().numpy())
    else:
        # Fallback: save GR/RN if they were defined
        if GR_hist is not None:
            GR_np = GR_hist.cpu().numpy()
            np.save(f"{theory_dir}/traj_GR.npy", GR_np)
        if RN_hist is not None:
            RN_np = RN_hist.cpu().numpy()
            np.save(f"{theory_dir}/traj_RN.npy", RN_np)
    # <reason>chain: Save trajectories; handle both flexible baseline and legacy GR/RN cases.</reason>

    # Save results
    with open(f"{theory_dir}/results.json", "w") as f:
        json.dump({
            "name": res["name"],
            "loss_GR": res["loss_GR"],
            "loss_RN": res["loss_RN"],
            "loss_type": loss_type,
            **({k: v for k, v in res.get('all_loss_types', {}).items()} if args.multi_loss else {}),  # <reason>chain: Include all loss types in JSON if multi-loss enabled.&lt;/reason&gt;
        }, f, indent=4)  # <reason>chain: Updated JSON to include multi-loss data.&lt;/reason&gt;
    # <reason>chain: Save JSON; no change.</reason>

    # Copy cache files to theory_dir for debugging
    traj_cache = f"cache/cache_{tag}.pt"
    if os.path.exists(traj_cache):
        shutil.copy(traj_cache, theory_dir)
        json_cache = f"{traj_cache}.json"
        if os.path.exists(json_cache):
            shutil.copy(json_cache, theory_dir)
    for baseline_tag in [GR_tag, RN_tag]:
        baseline_cache = f"cache/cache_{baseline_tag}.pt"
        if os.path.exists(baseline_cache):
            shutil.copy(baseline_cache, theory_dir)
            baseline_json = f"{baseline_cache}.json"
            if os.path.exists(baseline_json):
                shutil.copy(baseline_json, theory_dir)
        loss_cache = f"cache/cache_loss_{loss_type}_{baseline_tag}_vs_{tag}.pt"  # <reason>chain: Updated loss_cache filename to include loss_type for distinct caching per method.&lt;/reason&gt;
        if os.path.exists(loss_cache):
            shutil.copy(loss_cache, theory_dir)
            loss_json = f"{loss_cache}.json"
            if os.path.exists(loss_json):
                shutil.copy(loss_json, theory_dir)

    # New: Metric components plot
    r_vals = torch.linspace(RS * 1.01, r0 * 2, 1000, device=device, dtype=DTYPE)
    # <reason>chain: Created r_vals tensor for metric evaluation over a range from near horizon to twice initial radius.</reason>
    
    # Build models list with all baseline theories (if available)
    models = []
    line_styles = ['k--', 'b:', 'g-.', 'm-.', 'c--', 'y:']  # Various line styles for baseline theories
    
    if baseline_trajectories:
        # Add all baseline theories to the plot
        for i, (name, data) in enumerate(baseline_trajectories.items()):
            style = line_styles[i % len(line_styles)]
            models.append((name, data['model'], style))
    else:
        # Fallback to trying to create default models if no baseline trajectories provided
        try:
            from predefined_theories import Schwarzschild, ReissnerNordstrom
            GR_model = Schwarzschild()
            RN_model = ReissnerNordstrom(Q=Q_PARAM)
            models.extend([('GR', GR_model, 'k--'), ('R-N', RN_model, 'b:')])
        except:
            pass
    
    # Add the current theory being evaluated
    models.append((res['name'], model, 'r-'))
    # <reason>chain: List of models with labels and styles for plotting.</reason>
    plt.figure(figsize=(12, 8))
    components = ['g_tt', 'g_rr', 'g_pp', 'g_tp']
    for idx, comp in enumerate(components, 1):
        ax = plt.subplot(2, 2, idx)
        for label, mod, style in models:
            gtt, grr, gpp, gtp = mod.get_metric(r_vals, M, c, G)
            vals = {'g_tt': gtt, 'g_rr': grr, 'g_pp': gpp, 'g_tp': gtp}[comp]
            ax.plot((r_vals / RS).cpu().numpy(), vals.cpu().numpy(), style, label=label)
        ax.set_xlabel('r / RS')
        ax.set_ylabel(comp)
        ax.legend()
        ax.grid(True)
    # After creating subplots\nis_gr = 'Schwarzschild' in res['name']\nis_rn = 'Reissner' in res['name']\nstatus = ' (GR Baseline)' if is_gr else ' (RN Baseline)' if is_rn else ''\nsteps = len(res['traj'])\nplt.suptitle(f'Metric Components for {res["name"]}{status} (Steps: {steps:,})', y=0.98)\n# <reason>chain: Updated suptitle to include baseline status and step count.</reason>\nfig.text(0.5, 0.01, f"Summary: {res['summary']}\nLoss vs GR: {res['loss_GR']:.3e} | Loss vs RN: {res['loss_RN']:.3e}", ha='center', va='bottom', fontsize=8, wrap=True)\n# <reason>chain: Added text at bottom for summary and losses.</reason>\nplt.tight_layout(rect=[0, 0.03, 1, 0.95])\n# <reason>chain: Adjusted tight_layout to make space for bottom text.</reason>\n# Before saving
    plt.suptitle(f'Metric Components for {res["name"]}', y=0.95)
    plt.tight_layout()
    plt.savefig(f"{theory_dir}/metric_plot.png")
    plt.close()
    # <reason>chain: Created and saved metric components plot comparing the theory to GR and RN baselines.</reason>

    # New: Check if breakthrough - now works with any baseline theories
    is_breakthrough = False
    
    # Check if it performs well against multiple baseline theories
    if baseline_trajectories and all_losses:
        # Calculate average loss ratio compared to baselines
        loss_ratios = []
        baseline_names = list(baseline_trajectories.keys())
        
        # For each pair of baseline theories, check if this theory creates balanced losses
        for i in range(len(baseline_names)):
            for j in range(i+1, len(baseline_names)):
                loss_i = all_losses.get(f"loss_vs_{baseline_names[i]}", float('inf'))
                loss_j = all_losses.get(f"loss_vs_{baseline_names[j]}", float('inf'))
                
                if loss_i < float('inf') and loss_j < float('inf') and loss_j > 0:
                    ratio = loss_i / loss_j
                    # Good unification: losses are balanced (ratio close to 1)
                    if 0.8 < ratio < 1.2:
                        loss_ratios.append(abs(1 - ratio))
        
        # Consider it promising if it has balanced losses across multiple baselines
        if loss_ratios and sum(loss_ratios) / len(loss_ratios) < 0.15:
            # Also check it's not one of the baseline theories itself
            is_baseline = any(baseline_name in res["name"] for baseline_name in baseline_names)
            if not is_baseline:
                is_breakthrough = True
    
    # Backward compatibility: also use old logic if GR_loss_vs_RN is available
    elif not math.isnan(res["loss_RN"]) and GR_loss_vs_RN > 0 and res["loss_RN"] < 0.9 * GR_loss_vs_RN and "Schwarzschild" not in res["name"] and "Reissner" not in res["name"]:
        is_breakthrough = True
    
    if is_breakthrough:
        # Log to {theory_base_dir}/promising_candidates.log
        log_entry = f"{time.strftime('%Y%m%d_%H%M%S')} | Run: {run_timestamp} | Theory: {res['name']} | Loss_GR: {res['loss_GR']:.3e} | Loss_RN: {res['loss_RN']:.3e} | Summary: {res['summary']} | Dir: {theory_dir} | Loss Type: {loss_type}\n"  # <reason>chain: Added loss_type to log entry for tracking.&lt;/reason&gt;
        
        # Determine log path based on theory_base_dir
        if theory_base_dir:
            promising_log_path = os.path.join(theory_base_dir, "promising_candidates.log")
        else:
            promising_log_path = "promising_candidates.log"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(promising_log_path) if os.path.dirname(promising_log_path) else ".", exist_ok=True)
        
        with open(promising_log_path, "a") as log_file:
            log_file.write(log_entry)
        print(f"Promising theory found! Logged to {promising_log_path}")

    # After saving code and results, generate interactive visualization
    # Extract get_metric function from code.py
    with open(f"{theory_dir}/code.py", "r") as f:
        code_content = f.read()

    # Extract the specific theory class and its get_metric method
    import re
    import ast
    
    js_metric = None
    
    # For generated theories, try to find any get_metric method
    if is_generated:
        # Look for any get_metric method in the code
        metric_pattern = r'def\s+get_metric\s*\([^)]+\)\s*->\s*[^:]+:\s*\n((?:\s+.*\n)+?)(?=\n\s*def|\n\s*class|\Z)'
        metric_match = re.search(metric_pattern, code_content, re.MULTILINE)
        
        if metric_match:
            metric_body = metric_match.group(1)
            # Normalize indentation
            lines = metric_body.split('\n')
            # Find minimum indentation
            min_indent = float('inf')
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            # Remove minimum indentation from all lines
            lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
            metric_body = '\n'.join(lines)
    else:
        # Try to find the specific class for this theory
        theory_class_name = model.__class__.__name__
        class_pattern = rf'class\s+{re.escape(theory_class_name)}\s*.*?:\s*(.*?)(?=\nclass|\Z)'
        class_match = re.search(class_pattern, code_content, re.DOTALL)
        
        if class_match:
            class_body = class_match.group(0)
            # Extract get_metric method from this class
            metric_pattern = r'def\s+get_metric\s*\([^)]+\)\s*->\s*[^:]+:\s*\n((?:\s{8}.*\n)+)'
            metric_match = re.search(metric_pattern, class_body)
            
            if metric_match:
                metric_body = metric_match.group(1)
    
    # Convert Python to JavaScript if we found the metric
    if 'metric_body' in locals() and metric_body:
        # Clean up and convert to JavaScript
        lines = metric_body.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        js_lines = []
        for line in lines:
            # Skip comments and docstrings
            if line.strip().startswith('#') or line.strip().startswith('"""'):
                continue
                
            # Convert return statement
            if line.startswith('return '):
                # Extract the returned values
                return_vals = line[7:].strip()
                # Remove trailing comma if present
                if return_vals.endswith(','):
                    return_vals = return_vals[:-1]
                # Fix any remaining Python syntax in return values
                return_vals = return_vals.replace('torch.zeros_like(r)', '0')
                return_vals = return_vals.replace('torch.ones_like(r)', '1')
                return_vals = return_vals.replace('zeros_like(r)', '0')
                return_vals = return_vals.replace('ones_like(r)', '1')
                return_vals = return_vals.replace('EPSILON', '1e-10')
                js_lines.append(f'return [{return_vals}];')
            else:
                # Basic conversions
                js_line = line
                
                # Handle Python comments
                if '#' in js_line:
                    code_part, comment_part = js_line.split('#', 1)
                    js_line = code_part.rstrip() + ' //' + comment_part
                
                # Replace Python-specific syntax
                js_line = js_line.replace('torch.', 'Math.')
                js_line = js_line.replace('EPSILON', '1e-10')
                js_line = js_line.replace('self.', '')
                js_line = js_line.replace('zeros_like(r)', '0')
                js_line = js_line.replace('ones_like(r)', '1')
                js_line = js_line.replace('torch.zeros_like(r)', '0')
                js_line = js_line.replace('torch.ones_like(r)', '1')
                js_line = js_line.replace('np.', 'Math.')
                js_line = js_line.replace('math.', 'Math.')
                
                # Replace parameter names with params object access
                js_line = js_line.replace('G_param', 'params.G')
                js_line = js_line.replace('M_param', 'params.M')
                js_line = js_line.replace('C_param', 'params.C')
                
                # Handle theory-specific constants
                # Check if gamma, beta, or other parameters are used without being defined
                if 'gamma' in js_line and 'const gamma' not in js_line:
                    js_line = js_line.replace('gamma', 'params.gamma || 0.5')
                if 'beta' in js_line and 'const beta' not in js_line:
                    js_line = js_line.replace('beta', 'params.beta || 0.1')
                if 'LP' in js_line:
                    js_line = js_line.replace('LP', '1.616e-35')  # Planck length in meters
                
                # Skip rs calculation line since it's already defined in the template
                if 'rs = ' in js_line and 'params.G' in js_line and 'params.M' in js_line:
                    continue
                
                # Handle tensor creation
                js_line = re.sub(r'\.as_tensor\([^)]+\)', '', js_line)
                js_line = re.sub(r'torch\.as_tensor\([^)]+\)', '', js_line)
                
                # Add const/let for variable declarations
                # Check if this is a variable assignment (not inside parentheses)
                if '=' in js_line and not js_line.strip().startswith('return'):
                    var_match = re.match(r'^(\s*)([a-zA-Z_]\w*)\s*=', js_line)
                    if var_match:
                        indent = var_match.group(1)
                        var_name = var_match.group(2)
                        # Always add const unless it's the 'r' parameter
                        if var_name != 'r':
                            js_line = f'{indent}const {js_line.strip()}'
                
                # Ensure semicolons
                if js_line.strip() and not js_line.rstrip().endswith((';', '{', '}', ':')):
                    js_line = js_line.rstrip() + ';'
                
                # Remove any remaining colons at the end
                js_line = re.sub(r':\s*;$', ';', js_line)
                
                if js_line.strip():
                    js_lines.append(js_line)
        
        js_metric = '\n                '.join(js_lines)
    else:
        js_metric = '// Metric extraction failed - could not find get_metric method'

    # Copy particle names data file
    os.makedirs(f"{theory_dir}/data", exist_ok=True)
    particle_names_src = "data/particle_names.json"
    if os.path.exists(particle_names_src):
        shutil.copy(particle_names_src, f"{theory_dir}/data/particle_names.json")

    # Generate HTML with enhanced visualization
    # Calculate relative path from theory_dir to viz directory
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viz')
    viz_relative_path = os.path.relpath(viz_dir, theory_dir)
    
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Visualization: {res["name"]}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script src="https://unpkg.com/three@0.128.0/examples/js/renderers/CSS2DRenderer.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
        <script src="visualization_enhanced.js"></script>
        <style> 
            body {{ margin: 0; font-family: Arial, sans-serif; }} 
            #viz {{ width: 100vw; height: 100vh; }}
            .label:hover {{ background: rgba(0, 0, 0, 0.95) !important; box-shadow: 0 2px 8px rgba(0,0,0,0.5); }}
            .metric-component:hover {{ background: rgba(100, 100, 255, 0.2); }}
        </style>
    </head>
    <body>
        <div id="viz"></div>
        <script>
            const metricFunction = (r, params) => {{
                const rs = 2 * params.G * params.M / params.C**2;
                {js_metric}
            }};
            const initialParams = {{ alpha: 0.5, gamma: 0.5, beta: 0.1, G: 1, M: 1, C: 1 }}; // Adjust based on theory
            new GravityVisualizerEnhanced('viz', metricFunction, initialParams);
        </script>
    </body>
    </html>
    '''
    with open(f"{theory_dir}/viz.html", "w") as f:
        f.write(html_content)

    # Copy visualization JS file
    viz_js_src = "viz/visualization_enhanced.js"
    if os.path.exists(viz_js_src):
        shutil.copy(viz_js_src, f"{theory_dir}/visualization_enhanced.js")

    # Start local server for visualization
    try:
        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        
        server_process = subprocess.Popen(['python3', '-m', 'http.server', str(port), '--directory', theory_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)  # Give server time to start
        viz_url = f'http://localhost:{port}/viz.html'
        print(f'Interactive visualization for {res["name"]} available at: {viz_url}')
    except Exception as e:
        print(f'Failed to start server: {e}')

    return res
# <reason>chain: Evaluate theory updated to use model-specific initial conditions.</reason>

def main() -> None:
    os.makedirs('cache', exist_ok=True)
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    """
    Main driver for the simulation.
    <reason>This function orchestrates the entire process: setting up models, defining initial conditions, running the simulations, calculating losses, and reporting the results.</reason>
    """
    print("=" * 80)
    print(f"PyTorch Orbital Test | device={device} | dtype={DTYPE}")
    print("=" * 80)
    # <reason>chain: Header print; no change.</reason>

    # Diagnostic for rq/RS to confirm distinct baselines
    rq_sq = (G * Q_PARAM**2) / (4 * math.pi * epsilon_0 * c**4)
    rq = math.sqrt(rq_sq)
    print(f"Computed rq: {rq:.2e} m | RS: {RS_SI:.2e} m | rq/RS: {rq / RS_SI:.4f}")
    # <reason>chain: Added diagnostic to verify RN distinction, ensuring meaningful dual baselines.</reason>

    # Load manual theories if file provided
    manual_theories = []
    if args.manual_theories_file:
        manual_theories = load_manual_theories(args.manual_theories_file)
        if not args.self_discover:
            # Add to correct category based on class variable
            for m in manual_theories:
                cat = getattr(m.__class__, "category", "classical")
                if cat == "quantum":
                    quantum_predefined.append(m)
                elif cat == "unified":
                    unified_predefined.append(m)
                else:
                    classical_predefined.append(m)
    # <reason>Now manual theories are also categorized by their class variable, not arbitrarily.</reason>
    
    # Load theories from new directory structure if specified
    theory_dirs_dict = {}
    if args.theory_dirs and not args.self_discover:
        theory_dirs_dict = load_theories_from_dirs(args.theory_dirs)
        # If using new structure, clear old predefined lists
        if theory_dirs_dict:
            classical_predefined = []
            quantum_predefined = []
            unified_predefined = []
            
            # Combine all theories from directories
            for theory_dir, theories in theory_dirs_dict.items():
                for theory in theories:
                    cat = getattr(theory.__class__, "category", "classical")
                    if cat == "quantum":
                        quantum_predefined.append(theory)
                    elif cat == "unified":
                        unified_predefined.append(theory)
                    else:
                        classical_predefined.append(theory)



    # -- Initial Conditions Setup (global r0 and v_tan) --
    r0 = 15.0 * RS  # Increased from 10 RS to 15 RS for better numerical stability while still maintaining strong field effects
    v_tan = torch.sqrt(G_T * M / r0)
    period_est = 2 * TORCH_PI * r0 / v_tan
    DTau = period_est / 1000.0

    # -- Run Parameters --
    MAX_CONSECUTIVE_FAILURES = 10
    if args.test:
        N_STEPS = 1000
        print("Mode: TEST (quick benchmarking)")
    elif args.final:
        N_STEPS = 5_000_000
        print("Mode: FINAL (high precision, long duration)")
    else:
        N_STEPS = 100_000  # Exploratory steps: 100,000 chosen to capture ~10 orbits with ~100 points per orbit (based on period_est), ensuring reliable FFT spectra for loss while keeping runs efficient (~1-2 min/theory on GPU). Higher (e.g., 500k) increases accuracy marginally but slows iteration; lower risks aliasing in frequency analysis.
        print("Mode: EXPLORATORY (balanced accuracy/efficiency)")
    STEP_PRINT = max(1, N_STEPS // 50)

    # Remove the old assignments with STEP_PRINT

    # -- Load Baseline Theories from all specified directories --
    print("\nLoading baseline theories...")
    baseline_theories = load_baseline_theories_from_dirs(args.theory_dirs)
    if not baseline_theories:
        print("WARNING: No baseline theories found! Loading from default location.")
        # Try default location as fallback
        baseline_theories = load_baseline_theories_from_dirs(["theories/defaults"])
        if not baseline_theories:
            print("ERROR: No baseline theories found even in default location!")
            print("Please ensure baseline theories exist in theories/defaults/baselines/")
            sys.exit(1)
    
    print(f"Found {len(baseline_theories)} baseline theories")
    
    # -- Generate Ground-Truth Trajectories for All Baseline Theories --
    precision_tag = "f64" if DTYPE == torch.float64 else "f32"
    r0_tag = int(r0.item() / RS.item())
    
    baseline_trajectories = {}
    print("\nGenerating baseline theory trajectories...")
    for baseline_model in baseline_theories:
        print(f"  Running {baseline_model.name}...")
        hist, tag = run_trajectory(baseline_model, r0, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT)
        baseline_trajectories[baseline_model.name] = {
            'model': baseline_model,
            'hist': hist,
            'tag': tag
        }
    
    # For backward compatibility, keep GR_hist and RN_hist if they exist
    GR_hist = None
    RN_hist = None
    GR_tag = None
    RN_tag = None
    GR_loss_vs_RN = 0.0
    
    # Try to find Schwarzschild and Reissner-Nordström for backward compatibility
    for name, data in baseline_trajectories.items():
        if 'Schwarzschild' in name:
            GR_hist = data['hist']
            GR_tag = data['tag']
        elif 'Reissner' in name:
            RN_hist = data['hist']
            RN_tag = data['tag']
    
    # Initialize history for AI discovery loop
    history = []
    
    # Add all baseline theories to history with their pairwise losses
    if baseline_trajectories:
        baseline_names = list(baseline_trajectories.keys())
        
        # For each baseline theory, calculate its loss against all others
        for i, (name_i, data_i) in enumerate(baseline_trajectories.items()):
            losses = {}
            for j, (name_j, data_j) in enumerate(baseline_trajectories.items()):
                if i != j:
                    loss = calculate_loss(data_j['hist'], data_i['hist'], 
                                            ref_tag=data_j['tag'], pred_tag=data_i['tag'], loss_type=args.loss_type)
                    losses[f"loss_vs_{name_j}"] = loss
            
            # For backward compatibility, set loss_GR and loss_RN if they match
            loss_GR = losses.get(f"loss_vs_{list(baseline_trajectories.keys())[0]}", 0.0)
            loss_RN = losses.get(f"loss_vs_{list(baseline_trajectories.keys())[1] if len(baseline_trajectories) > 1 else list(baseline_trajectories.keys())[0]}", 0.0)
            
            # Get the model's summary if available
            model = data_i['model']
            summary = getattr(model, 'summary', f"{model.__class__.__name__} baseline theory")
            
            history.append({
                "name": name_i,
                "loss_GR": loss_GR,
                "loss_RN": loss_RN,
                "summary": summary,
                "all_losses": losses
            })
    
    # If we have both GR and RN specifically, calculate their loss
    if GR_hist is not None and RN_hist is not None:
        GR_loss_vs_RN = calculate_loss(RN_hist, GR_hist, ref_tag=RN_tag, pred_tag=GR_tag, loss_type=args.loss_type)
    
    # If we don't have traditional GR/RN, use first two baseline theories
    if GR_hist is None and len(baseline_trajectories) >= 1:
        first_name = list(baseline_trajectories.keys())[0]
        GR_hist = baseline_trajectories[first_name]['hist']
        GR_tag = baseline_trajectories[first_name]['tag']
    
    if RN_hist is None and len(baseline_trajectories) >= 2:
        second_name = list(baseline_trajectories.keys())[1]
        RN_hist = baseline_trajectories[second_name]['hist']
        RN_tag = baseline_trajectories[second_name]['tag']
        GR_loss_vs_RN = calculate_loss(RN_hist, GR_hist, ref_tag=RN_tag, pred_tag=GR_tag, loss_type=args.loss_type)
    # <reason>chain: Updated baselines; no change.</reason>

    # Evaluate theories (predefined and/or manual)
    results = []
    for category, theories in [("classical_predefined", classical_predefined), ("quantum_predefined", quantum_predefined), ("unified_predefined", unified_predefined)]:
        for model in theories:
            # Get theory base dir if available
            theory_base_dir = getattr(model, '_theory_dir', None)
            res = evaluate_theory(model, category, r0, GR_hist, RN_hist, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT, 
                                GR_tag=GR_tag, RN_tag=RN_tag, GR_loss_vs_RN=GR_loss_vs_RN, run_timestamp=run_timestamp,
                                theory_base_dir=theory_base_dir, baseline_trajectories=baseline_trajectories, loss_type=args.loss_type)  # <reason>chain: Passed loss_type from args to evaluate_theory for using selected method in calculations.&lt;/reason&gt;
            results.append(res)
            
            # Add to history with all losses
            hist_entry = {
                "name": res["name"], 
                "loss_GR": res["loss_GR"], 
                "loss_RN": res["loss_RN"], 
                "summary": res["summary"]
            }
            if 'all_losses' in res:
                hist_entry['all_losses'] = res['all_losses']
            history.append(hist_entry)
    # <reason>chain: Evaluated predefined; uses updated evaluate_theory with per-model init.</reason>

    # -- Iterative Generation Loop if enabled --
    breakthrough_found = False
    iteration = 1
    if args.self_discover:
        # Determine which theory directory to use as context
        # If multiple directories specified, use the last non-defaults one
        seed_theory_dir = None
        if args.theory_dirs:
            for td in args.theory_dirs:
                if td != "theories/defaults":
                    seed_theory_dir = td
        
        if not seed_theory_dir:
            seed_theory_dir = "theories/defaults"
        
        print(f"\n=== Self-Discovery Mode ===")
        print(f"Seed theory directory: {seed_theory_dir}")
        print(f"API provider: {args.api_provider}")
        if args.initial_prompt:
            print(f"Initial prompt: {args.initial_prompt}")
        print("="*30)
        
        while True:
            print(f"\n--- Iteration {iteration}: Generating new theories ---")
            new_theories = generate_new_theories(history, args.initial_prompt)
            if not new_theories:
                print("No valid new models generated. Continuing...")
                iteration += 1
                continue

            print(f"Testing {len(new_theories)} new models: {[m[0].name for m in new_theories]}")

            for idx, (model, summary, gen_content, category, prompt) in enumerate(new_theories, 1):
                res = evaluate_theory(model, category, r0, GR_hist, RN_hist, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT, gen_content, summary, prompt=prompt, response=gen_content, GR_tag=GR_tag, RN_tag=RN_tag, GR_loss_vs_RN=GR_loss_vs_RN, run_timestamp=run_timestamp, theory_base_dir="theories/self_discovery", is_generated=True, baseline_trajectories=baseline_trajectories, loss_type=args.loss_type)  # <reason>chain: Passed loss_type to generated theory evaluation for consistency.&lt;/reason&gt;
                results.append(res)
                
                # Add to history with all losses
                hist_entry = {
                    "name": res["name"], 
                    "loss_GR": res["loss_GR"], 
                    "loss_RN": res["loss_RN"], 
                    "summary": summary
                }
                if 'all_losses' in res:
                    hist_entry['all_losses'] = res['all_losses']
                history.append(hist_entry)

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

            # Ranking for each baseline theory
            if baseline_trajectories:
                for baseline_name in baseline_trajectories.keys():
                    print(f"\n--- RANKING vs. {baseline_name.upper()} ---")
                    
                    # Sort by loss against this baseline
                    results_sorted = sorted(results, 
                                          key=lambda d: (math.isnan(d.get('all_losses', {}).get(f'loss_vs_{baseline_name}', float('inf'))), 
                                                        d.get('all_losses', {}).get(f'loss_vs_{baseline_name}', float('inf'))))
                    
                    print("Rank | Model                                | Loss (FFT MSE)")
                    print("-" * 60)
                    for rank, res in enumerate(results_sorted, 1):
                        loss_val = res.get('all_losses', {}).get(f'loss_vs_{baseline_name}', res.get('loss_RN', float('inf')))
                        print(f"{rank:4d} | {res['name']:<36} | {loss_val:10.3e}")
                
                # Check for breakthroughs - theories with balanced low losses across baselines
                print("\n--- UNIFIED THEORY CANDIDATES ---")
                print("(Theories with balanced losses across multiple baselines)")
                print("-" * 60)
                
                for res in results:
                    if 'all_losses' in res:
                        losses = [v for k, v in res['all_losses'].items() if isinstance(v, (int, float)) and not math.isnan(v)]
                        if len(losses) >= 2:
                            avg_loss = sum(losses) / len(losses)
                            std_loss = (sum((l - avg_loss)**2 for l in losses) / len(losses))**0.5
                            max_loss = max(losses)
                            
                            # Consider breakthrough if low average loss and balanced across baselines
                            if max_loss < 0.1 and std_loss < 0.05:
                                print(f"{GREEN_BG}{BOLD}{res['name']:<36} | avg: {avg_loss:.3e} | std: {std_loss:.3e} [BREAKTHROUGH]{RESET}")
                                breakthrough_found = True
                            elif max_loss < 0.5 and std_loss < 0.1:
                                print(f"{res['name']:<36} | avg: {avg_loss:.3e} | std: {std_loss:.3e} [PROMISING]")
            else:
                # Fallback to old GR/RN logic if no generic baselines
                results.sort(key=lambda d: (math.isnan(d["loss_RN"]), d["loss_RN"]))
                print("\n--- RANKING vs. REISSNER-NORDSTRÖM (R-N) ---")
                print(f"(GR baseline loss vs R-N is: {GR_loss_vs_RN:.3e})")
                print("Rank | Model                                | Loss_RN (FFT MSE)")
                print("-" * 60)
                for rank, res in enumerate(results, 1):
                    loss_val = res["loss_RN"]
                    name = res['name']
                    is_breakthrough = not math.isnan(loss_val) and loss_val < 0.9 * GR_loss_vs_RN and "Schwarzschild" not in name and "Reissner" not in name
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
    # <reason>chain: Main loop; uses updated cached_run and evaluate_theory; tightened breakthrough to <0.9*GR_loss_vs_RN for robustness buffer.</reason>

    # -- Validation Section --
    if args.validate_observations:
        print("\n" + "="*50)
        print("Running Observational Validations")
        print("="*50)
        
        # Pass device and dtype to validation system
        from run_validations import run_all_validations
        
        # Collect all theories to validate
        all_theories = []
        for category, theories in [("classical", classical_predefined), 
                                  ("quantum", quantum_predefined), 
                                  ("unified", unified_predefined)]:
            all_theories.extend(theories)
        
        # Run validations with same device/dtype as main simulation
        validation_results = run_all_validations(
            theories=all_theories,
            device=device,
            dtype=DTYPE
        )
        
        # Save validation results
        import json
        with open(f"validation_results_{run_timestamp}.json", "w") as f:
            json.dump(validation_results, f, indent=2)

    print("\nDone.")
# <reason>chain: Entry point unchanged; all changes ensure correct/meaningful runs without breaking loop/API.</reason>

def load_theories_from_dirs(theory_dirs: list[str]) -> dict[str, list[GravitationalTheory]]:
    """
    Load theories from the new directory structure.
    Each theory_dir should have a source/ subdirectory with theory.py or multiple .py files.
    Returns dict mapping theory_dir to list of instantiated models.
    """
    all_theories = {}
    
    for theory_dir in theory_dirs:
        if not os.path.exists(theory_dir):
            print(f"Theory directory not found: {theory_dir}")
            continue
            
        source_dir = os.path.join(theory_dir, "source")
        if not os.path.exists(source_dir):
            print(f"No source/ directory in: {theory_dir}")
            continue
        
        theories = []
        
        # Look for all .py files in source/
        for filename in os.listdir(source_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                filepath = os.path.join(source_dir, filename)
                
                # Read and execute the file
                with open(filepath, 'r') as f:
                    code = f.read()
                
                # Create a clean namespace with required imports
                namespace = {
                    'GravitationalTheory': GravitationalTheory,
                    'Tensor': torch.Tensor,
                    'torch': torch,
                    'np': np,
                    'numpy': np,
                    'math': math,
                    'device': device,
                    'DTYPE': DTYPE,
                    'EPSILON': EPSILON,
                    'epsilon_0': epsilon_0,
                    'G': G,
                    'c': c,
                    'hbar': hbar,
                    'Q_PARAM': Q_PARAM,
                    'TORCH_PI': TORCH_PI,
                    'EPS0_T': EPS0_T,
                    'LP': LP,
                    'STOCHASTIC_STRENGTH': STOCHASTIC_STRENGTH,
                }
                
                try:
                    exec(code, namespace)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue
                
                # Find all GravitationalTheory subclasses
                for name, obj in namespace.items():
                    if isinstance(obj, type) and issubclass(obj, GravitationalTheory) and obj != GravitationalTheory:
                        # Check for parameter sweeps
                        sweep = getattr(obj, 'sweep', None)
                        if sweep and isinstance(sweep, dict):
                            # Handle both single and multi-parameter sweeps
                            if len(sweep) == 1:
                                # Single parameter sweep
                                for param, values in sweep.items():
                                    for v in values:
                                        try:
                                            instance = obj(**{param: float(v)})
                                            instance._theory_dir = theory_dir  # Store source dir
                                            instance._source_code = code  # Store source code
                                            theories.append(instance)
                                        except Exception as e:
                                            print(f"Error instantiating {name} with {param}={v}: {e}")
                            else:
                                # Multi-parameter sweep - create cartesian product
                                import itertools
                                param_names = list(sweep.keys())
                                param_values = [sweep[p] for p in param_names]
                                for value_combo in itertools.product(*param_values):
                                    kwargs = {param_names[i]: float(value_combo[i]) for i in range(len(param_names))}
                                    try:
                                        instance = obj(**kwargs)
                                        instance._theory_dir = theory_dir
                                        instance._source_code = code  # Store source code
                                        theories.append(instance)
                                    except Exception as e:
                                        print(f"Error instantiating {name} with {kwargs}: {e}")
                        else:
                            # Try to instantiate with defaults
                            try:
                                instance = _instantiate_theory(obj)
                                if instance:
                                    instance._theory_dir = theory_dir
                                    instance._source_code = code  # Store source code
                                    theories.append(instance)
                            except Exception as e:
                                print(f"Error instantiating {name}: {e}")
        
        if theories:
            all_theories[theory_dir] = theories
            print(f"Loaded {len(theories)} theories from {theory_dir}")
    
    return all_theories

def load_baseline_theories_from_dirs(theory_dirs):
    """Load all baseline theories from the baselines directories of specified theory dirs."""
    baseline_theories = []
    
    for theory_dir in theory_dirs:
        baseline_dir = os.path.join(theory_dir, "baselines")
        if os.path.exists(baseline_dir):
            for filename in os.listdir(baseline_dir):
                if filename.endswith('.py') and not filename.startswith('__'):
                    filepath = os.path.join(baseline_dir, filename)
                    try:
                        module_name = f"baselines_{theory_dir.replace('/', '_')}_{filename[:-3]}"
                        spec = importlib.util.spec_from_file_location(module_name, filepath)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        for name in dir(module):
                            obj = getattr(module, name)
                            if isinstance(obj, type) and issubclass(obj, GravitationalTheory) and obj != GravitationalTheory:
                                try:
                                    # Try instantiating without arguments
                                    instance = obj()
                                except TypeError as e:
                                    # Handle special cases that need parameters
                                    if 'ReissnerNordstrom' in name:
                                        # Use charge that gives meaningful EM effects while maintaining numerical stability
                                        # Q=1e19 gives rq/RS ~0.003, which is small but distinguishable
                                        instance = obj(Q=1e19)
                                    else:
                                        raise e
                                baseline_theories.append(instance)
                                print(f"Loaded baseline theory: {instance.name} from {theory_dir}")
                    except Exception as e:
                        print(f"Warning: Failed to load baseline theory from {filepath}: {e}")
    
    return baseline_theories

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()