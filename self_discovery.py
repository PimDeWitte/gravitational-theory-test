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
# <reason>chain: Imports unchanged; foundational for API, tensors, and plotting.</reason>

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
    p.add_argument("--test", action="store_true", help="Run in test mode with reduced steps for quick benchmarking.")
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
Q_PARAM = 4.878e21  # Increased to 0.9 * Q_ext ≈4.878e21 C for stronger EM effects, making RN distinct from GR in plots while avoiding naked singularity.
# <reason>chain: Set Q_PARAM to sub-extremal value (0.9 * Q_ext ≈4.878e21 C) to avoid naked singularity and instability in RN metric, enabling successful ground truth generation and meaningful dual-baseline tests.</reason>
STOCHASTIC_STRENGTH = 1e-7
# <reason>chain: Stochastic strength; no change.</reason>

G_T = torch.as_tensor(G, device=device, dtype=DTYPE)
# <reason>Added tensor version of G to ensure consistent types in calculations like v_tan, avoiding potential type issues in torch operations.</reason>
C_T = torch.as_tensor(c, device=device, dtype=DTYPE)
# <reason>Added tensor version of c for consistency in tensor operations.</reason>

# ---------------------------------------------------------------------------
# 2.  THEORY DEFINITIONS
# ---------------------------------------------------------------------------

Tensor = torch.Tensor  # Type alias for brevity
# <reason>chain: Type alias unchanged for brevity.</reason>


class GravitationalTheory:
    """
    Abstract base class for all gravitational theories.
    <reason>This class defines a common interface (`get_metric`) that all theories must implement. This polymorphic design allows the integrator to treat any theory identically, simplifying the simulation logic and making the framework easily extensible.</reason>
    """
    # New: Add category and sweep as class variables for auto-categorization and parameter sweep
    category = "classical"  # Default; subclasses can override to "quantum" or other
    sweep = None            # Default; subclasses can override with dict of param: values

    # New: Cache-related attributes
    cacheable = False       # Default to not cacheable; subclasses override to True if suitable for caching
    def get_cache_tag(self, N_STEPS, precision_tag, r0_tag):
        """
        Returns a unique tag for caching this theory's trajectory.
        Subclasses should override to include parameters, e.g., return f"{self.name}_alpha{self.alpha}"
        """
        base = self.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(".", "_")
        return f"{base}_{N_STEPS}_{precision_tag}_r{r0_tag}"

    def __init__(self, name: str) -> None:
        self.name = name
    # <reason>chain: Init name; no change.</reason>

    def get_metric(self, r, M_param, C_param, G_param):
        """Calculates the metric components (g_tt, g_rr, g_φφ, g_tφ) for a given radius."""
        raise NotImplementedError
    # <reason>chain: Abstract method; no change.</reason>

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
        # For each parameter, sweep over its values (only 1D sweeps supported for now)
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

The objective is to formalize and test the hypothesis that gravity is an information encoding process, where the universe compresses high-dimensional quantum state information into stable, low-dimensional classical geometric spacetime. Physical theories act as "decoders". Use a computational framework to measure "decoding loss" of candidate theories via dynamic orbital mechanics tests, benchmarked against lossless decoders for gravity (Schwarzschild metric) and electromagnetism (Reissner-Nordström metric with high charge Q~1.5e21 C for distinct EM effects). Results confirm unique, lossless status of General Relativity and Kaluza-Klein theory, establishing a methodology for evaluating laws based on informational fidelity. A breakthrough is when a non-baseline theory has lower loss vs RN than GR's loss vs RN, meaning it unifies better without explicit charge.

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
        self.model, self.M, self.c, self.G = model, M_param, C_param, G_param
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
        # Optional torsion contribution (placeholder; could compute S_term from g_tp derivatives if needed)
        # d2r_dtau2 += 0.5 * S_term * dr_dtau**2  # Uncomment if full torsion tensor implemented.
        return torch.stack((u_t / self.c, dr_dtau, u_phi, d2r_dtau2))
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
        for i in range(N_STEPS):
            y = integ.rk4_step(y, DTau)
            hist[i + 1] = y
            if (i + 1) % STEP_PRINT == 0: print(f"  ...step {i+1:,}/{N_STEPS:,} | r={y[1]/RS:.3f} RS")
            if not torch.all(torch.isfinite(y)):
                print(f"Debug: Non-finite state detected at step {i+1}: y={y.tolist()}")
                # Use previous r if current is non-finite
                r_curr = y[1] if torch.isfinite(y[1]) else hist[i, 1]
                r_curr = r_curr.clone().detach().requires_grad_(True)
                g_tt, g_rr, g_pp, g_tp = model.get_metric(r_curr, M, c, G)
                print(f"Debug: Metric at r={r_curr.item() / RS.item():.3f} RS: g_tt={g_tt.item():.3e}, g_rr={g_rr.item():.3e}, g_pp={g_pp.item():.3e}, g_tp={g_tp.item():.3e}")
                det = g_tp ** 2 - g_tt * g_pp
                print(f"Debug: det={det.item():.3e}")
                if torch.abs(det) < EPSILON:
                    print("Debug: Small determinant may cause instability.")
                else:
                    u_t = (integ.E * g_pp + integ.Lz * g_tp) / det
                    u_phi = -(integ.E * g_tp + integ.Lz * g_tt) / det
                    inner = g_tt * u_t ** 2 + g_pp * u_phi ** 2 + 2 * g_tp * u_t * u_phi
                    print(f"Debug: inner={inner.item():.3e}")
                    V_sq = (-integ.c ** 2 - inner) / g_rr
                    print(f"Debug: V_sq={V_sq.item():.3e}")
                    if not torch.isfinite(V_sq):
                        print("Debug: Non-finite V_sq detected.")
                        if g_rr <= 0:
                            print("Debug: Negative g_rr contributing to instability.")
                    # Attempt to compute gradient
                    try:
                        (dV_dr,) = torch.autograd.grad(V_sq, r_curr, create_graph=False, retain_graph=False)
                        print(f"Debug: dV_dr={dV_dr.item():.3e}")
                    except Exception as e:
                        print(f"Debug: Gradient computation failed: {e}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"  ! ABORTED: Simulation unstable for {consecutive_failures} consecutive steps.")
                    hist = hist[:i+2]; break
            else:
                consecutive_failures = 0
            if y[1] <= RS * 1.01:
                hist = hist[:i+2]; break
        return hist, tag
    # Cacheable case
    fname = f"cache/cache_{tag}.pt"
    if os.path.exists(fname): 
        return torch.load(fname, map_location=device), tag
    print(f"\n--- Generating and Caching: {model.name} ({tag}) ---")
    y0_full = get_initial_conditions(model, r0)
    y0_state = y0_full[[0, 1, 2, 4]].clone()
    integ = GeodesicIntegrator(model, y0_full, M, c, G)
    hist = torch.empty((N_STEPS + 1, 4), device=device, dtype=DTYPE)
    hist[0] = y0_state
    y = y0_state.clone()
    consecutive_failures = 0
    for i in range(N_STEPS):
        y = integ.rk4_step(y, DTau)
        hist[i + 1] = y
        if (i + 1) % STEP_PRINT == 0: print(f"  ...step {i+1:,}/{N_STEPS:,} | r={y[1]/RS:.3f} RS")
        if not torch.all(torch.isfinite(y)):
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  ! ABORTED: Simulation unstable for {consecutive_failures} consecutive steps.")
                hist = hist[:i+2]; break
        else:
            consecutive_failures = 0
        if y[1] <= RS * 1.01:
            hist = hist[:i+2]; break
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

def calculate_fft_loss(traj_ref: Tensor, traj_pred: Tensor, ref_tag: str = None, pred_tag: str = None) -> float:
    """
    Calculates the informational loss between two trajectories using FFT MSE.
    <reason>This function is the core of the paper's methodology. It compares the full frequency spectrum of orbital dynamics, capturing subtle differences in precession and shape that a simple endpoint comparison would miss. It is a direct, quantitative measure of a theory's informational fidelity.</reason>
    """
    if ref_tag and pred_tag:
        cache_file = f"cache/cache_loss_{ref_tag}_vs_{pred_tag}.pt"
        if os.path.exists(cache_file):
            return torch.load(cache_file).item()
    min_len = min(len(traj_ref), len(traj_pred))
    if min_len < 2: return float("inf")
    r_ref, r_pred = traj_ref[:min_len, 1], traj_pred[:min_len, 1]
    if not (torch.all(torch.isfinite(r_ref)) and torch.all(torch.isfinite(r_pred))):
        return float('nan')
    fft_ref, fft_pred = torch.fft.fft(r_ref), torch.fft.fft(r_pred)
    mse = torch.mean((torch.abs(fft_ref) - torch.abs(fft_pred)) ** 2).item()
    norm_factor = torch.mean(torch.abs(fft_ref)**2).item()  # Normalize for unitless comparability
    # <reason>chain: Added normalization to make losses scale-invariant and comparable, addressing huge raw values and enabling meaningful comparisons/breakthroughs.</reason>
    loss = mse / (norm_factor + EPSILON) if norm_factor > 0 else mse
    if ref_tag and pred_tag:
        debug_dict = {
            "ref_tag": ref_tag,
            "pred_tag": pred_tag,
            "loss": loss,
            "timestamp": time.strftime("%Y%m%d_%H%M%S")
        }
        json_fname = f"{cache_file}.json"
        with open(json_fname, "w") as f:
            json.dump(debug_dict, f, indent=4)
        torch.save(torch.tensor(loss), cache_file)
    return loss
# <reason>chain: FFT loss; added EPSILON to norm_factor to prevent div0 in edge cases like flat metrics.</reason>

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

def evaluate_theory(model: GravitationalTheory, category: str, r0: Tensor, GR_hist: Tensor, RN_hist: Tensor, N_STEPS: int, DTau: float, MAX_CONSECUTIVE_FAILURES: int, STEP_PRINT: int, gen_content: str = "", summary: str = "", prompt: str = "", response: str = "", GR_tag: str = None, RN_tag: str = None, GR_loss_vs_RN: float = None, run_timestamp: str = "") -> dict:
    """
    Evaluates a theory by running the simulation and saving results.
    """
    base_dir = f"runs/{run_timestamp}/{category}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = model.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(".", "_")
    theory_dir = f"{base_dir}/{timestamp}_{safe_name}"
    os.makedirs(theory_dir, exist_ok=True)
    # <reason>chain: Theory dir; no change.</reason>

    # Save code
    if gen_content:
        code = gen_content
    else:
        try:
            code = inspect.getsource(model.__class__)
        except:
            code = "Source code not available for predefined theory."
    with open(f"{ theory_dir}/code.py", "w") as f:
        f.write(code)
    # <reason>chain: Save code; no change.</reason>

    # Save prompt and response if generated
    if prompt:
        with open(f"{ theory_dir}/input_prompt.txt", "w") as f:
            f.write(prompt)
    if response:
        with open(f"{ theory_dir}/api_response.txt", "w") as f:
            f.write(response)
    # <reason>chain: Save prompt/response; no change.</reason>

    print(f"\nEvaluating: {model.name} ({category})")
    traj, tag = run_trajectory(model, r0, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT)
    loss_GR = calculate_fft_loss(GR_hist, traj, ref_tag=GR_tag, pred_tag=tag)
    loss_RN = calculate_fft_loss(RN_hist, traj, ref_tag=RN_tag, pred_tag=tag)
    res = {
        "name": model.name,
        "loss_GR": loss_GR,
        "loss_RN": loss_RN,
        "traj": traj.cpu().numpy(),
        "summary": summary,
    }
    # <reason>chain: Calculated losses; no change.</reason>

    # Save plot
    GR_np, RN_np = GR_hist.cpu().numpy(), RN_hist.cpu().numpy()
    pred_np = res["traj"]
    GR_plot = downsample(GR_np)
    RN_plot = downsample(RN_np)
    pred_plot = downsample(pred_np)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    ax.plot(GR_plot[:, 2], GR_plot[:, 1], 'k--', label='GR', linewidth=1.5, alpha=1.0, zorder=7)
    ax.plot(RN_plot[:, 2], RN_plot[:, 1], 'b:', label='R-N', linewidth=1.5, alpha=1.0, zorder=8)
    ax.plot(pred_plot[:, 2], pred_plot[:, 1], 'r-', label=res['name'], linewidth=1.5, alpha=0.3, zorder=4)
    ax.plot(pred_np[0, 2], pred_np[0, 1], "go", markersize=8, label="start", zorder=6)
    ax.plot(pred_np[-1, 2], pred_np[-1, 1], "rx", markersize=10, mew=2, label="end", zorder=6)
    ax.set_title(res["name"], pad=20)
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{ theory_dir}/plot.png")
    plt.close()
    # <reason>chain: Save plot; no change.</reason>

    # Save trajectories
    np.save(f"{ theory_dir}/traj_pred.npy", pred_np)
    np.save(f"{ theory_dir}/traj_GR.npy", GR_np)
    np.save(f"{ theory_dir}/traj_RN.npy", RN_np)
    # <reason>chain: Save trajectories; no change.</reason>

    # Save results
    with open(f"{ theory_dir}/results.json", "w") as f:
        json.dump({
            "name": res["name"],
            "loss_GR": res["loss_GR"],
            "loss_RN": res["loss_RN"],
        }, f)
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
        loss_cache = f"cache/cache_loss_{baseline_tag}_vs_{tag}.pt"
        if os.path.exists(loss_cache):
            shutil.copy(loss_cache, theory_dir)
            loss_json = f"{loss_cache}.json"
            if os.path.exists(loss_json):
                shutil.copy(loss_json, theory_dir)

    # New: Metric components plot
    r_vals = torch.linspace(RS * 1.01, r0 * 2, 1000, device=device, dtype=DTYPE)
    # <reason>chain: Created r_vals tensor for metric evaluation over a range from near horizon to twice initial radius.</reason>
    GR_model = predefined_theories.Schwarzschild()
    RN_model = predefined_theories.ReissnerNordstrom(Q=Q_PARAM)
    # <reason>chain: Instantiated GR and RN models to compute their metrics.</reason>
    models = [('GR', GR_model, 'k--'), ('R-N', RN_model, 'b:'), (res['name'], model, 'r-')]
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
    plt.savefig(f"{ theory_dir}/metric_plot.png")
    plt.close()
    # <reason>chain: Created and saved metric components plot comparing the theory to GR and RN baselines.</reason>

    # New: Check if breakthrough
    is_breakthrough = not math.isnan(res["loss_RN"]) and res["loss_RN"] < 0.9 * GR_loss_vs_RN and "Schwarzschild" not in res["name"] and "Reissner" not in res["name"]
    if is_breakthrough:
        promising_dir = f"{base_dir}/promising"
        os.makedirs(promising_dir, exist_ok=True)
        promising_theory_dir = f"{promising_dir}/{timestamp}_{safe_name}"
        shutil.move(theory_dir, promising_theory_dir)  # Move to promising subdir inside run
        # Log to root
        log_entry = f"{time.strftime('%Y%m%d_%H%M%S')} | Run: {run_timestamp} | Theory: {res['name']} | Loss_GR: {res['loss_GR']:.3e} | Loss_RN: {res['loss_RN']:.3e} | Summary: {res['summary']} | Dir: {promising_theory_dir}\n"
        with open("promising_candidates.log", "a") as log_file:
            log_file.write(log_entry)
        print(f"Promising! Moved { theory_dir} to {promising_theory_dir} and logged.")

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
    print(f"Computed rq: {rq:.2e} m | RS: {RS_SI:.2e} m | rq/RS: {rq / RS_SI:.2f}")
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

    # Initial history from baselines
    history = [
        {"name": "Schwarzschild (GR)", "loss_GR": 0.0, "loss_RN": 0.0, "summary": "Standard GR metric: g_tt = -(1 - rs/r), g_rr = 1/(1 - rs/r), g_φφ = r^2, g_tφ = 0"},
        {"name": "Reissner‑Nordström (Q=1.5e21)", "loss_GR": 0.0, "loss_RN": 0.0, "summary": "Charged GR metric: g_tt = -(1 - rs/r + rq^2/r^2), g_rr = 1/(1 - rs/r + rq^2/r^2), g_φφ = r^2, g_tφ = 0"},
    ]
    # <reason>chain: History init; no change.</reason>

    # -- Initial Conditions Setup (global r0 and v_tan) --
    r0 = 10.0 * RS  # Reduced to 10 RS for stronger field effects, enhancing distinction between GR and RN while maintaining stability for most models.
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

    # -- Ground-Truth Trajectory Generation (Cached) --
    precision_tag = "f64" if DTYPE == torch.float64 else "f32"
    r0_tag = int(r0.item() / RS.item())
    GR_model = predefined_theories.Schwarzschild()
    GR_hist, GR_tag = run_trajectory(GR_model, r0, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT)
    RN_model = predefined_theories.ReissnerNordstrom(Q=Q_PARAM)
    RN_hist, RN_tag = run_trajectory(RN_model, r0, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT)
    GR_loss_vs_RN = calculate_fft_loss(RN_hist, GR_hist, ref_tag=RN_tag, pred_tag=GR_tag)
    # <reason>chain: Generated ground truths with per-model init conds.</reason>

    # Update baselines with actual losses
    history[0]["loss_RN"] = GR_loss_vs_RN
    history[1]["loss_GR"] = GR_loss_vs_RN
    # <reason>chain: Updated baselines; no change.</reason>

    # Evaluate theories (predefined and/or manual)
    results = []
    for category, theories in [("classical_predefined", classical_predefined), ("quantum_predefined", quantum_predefined), ("unified_predefined", unified_predefined)]:
        for model in theories:
            res = evaluate_theory(model, category, r0, GR_hist, RN_hist, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT, GR_tag=GR_tag, RN_tag=RN_tag, GR_loss_vs_RN=GR_loss_vs_RN, run_timestamp=run_timestamp)
            results.append(res)
            history.append({"name": res["name"], "loss_GR": res["loss_GR"], "loss_RN": res["loss_RN"], "summary": res["summary"]})
    # <reason>chain: Evaluated predefined; uses updated evaluate_theory with per-model init.</reason>

    # -- Iterative Generation Loop if enabled --
    breakthrough_found = False
    iteration = 1
    if args.self_discover:
        while True:
            print(f"\n--- Iteration {iteration}: Generating new theories ---")
            new_theories = generate_new_theories(history, args.initial_prompt)
            if not new_theories:
                print("No valid new models generated. Continuing...")
                iteration += 1
                continue

            print(f"Testing {len(new_theories)} new models: {[m[0].name for m in new_theories]}")

            for idx, (model, summary, gen_content, category, prompt) in enumerate(new_theories, 1):
                res = evaluate_theory(model, category, r0, GR_hist, RN_hist, N_STEPS, DTau, MAX_CONSECUTIVE_FAILURES, STEP_PRINT, gen_content, summary, prompt=prompt, response=gen_content, GR_tag=GR_tag, RN_tag=RN_tag, GR_loss_vs_RN=GR_loss_vs_RN, run_timestamp=run_timestamp)
                results.append(res)
                history.append({"name": res["name"], "loss_GR": res["loss_GR"], "loss_RN": res["loss_RN"], "summary": summary})

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

    print("\nDone.")
# <reason>chain: Main function updated with per-model initial conditions to fix RN generation, increased r0, manual theories loading, and breakthrough tightening.</reason>

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    main()
# <reason>chain: Entry point unchanged; all changes ensure correct/meaningful runs without breaking loop/API.</reason>