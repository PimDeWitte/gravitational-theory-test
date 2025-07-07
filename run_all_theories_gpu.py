# run_all_theories_gpu.py

# --- GPU MODIFICATION: Import CuPy as the primary array library. ---
import cupy as cp
import time
from scipy.constants import G, c, k_b, hbar, Lambda, epsilon_0
# Import the GPU-accelerated model.
from gravity_compression_gpu import CompressionModel

# --- HPC OPTIMIZATION: High-density grid for precision runs on a cluster. ---
DTYPE = cp.float32 # Use 32-bit floats for better performance on many GPUs.
NUM_POINTS = 1_000_000 # Increase grid density for a high-precision run.

# --- Global Simulation Parameters (Initialized directly on the GPU) ---
M_SOLAR = 1.989e30
M = cp.array(M_SOLAR, dtype=DTYPE)

# --- Pre-calculated Constants on the GPU ---
RS = cp.array((2 * G * M) / (c**2), dtype=DTYPE)
J_PARAM = cp.array(1e42, dtype=DTYPE)
Q_PARAM = cp.array(1e12, dtype=DTYPE)
OBSERVER_ENERGY = cp.array(1e9, dtype=DTYPE)

# --- Simulation Grid created on the GPU ---
r = cp.linspace(RS * 0.1, RS * 10, NUM_POINTS, dtype=DTYPE)

# --- All 17 Model Definitions ---
# These functions will now automatically operate on CuPy arrays.
# (Functions omitted for brevity, paste all 17 from the previous response here)
def compression_schwarzschild(r, M, **kwargs): return -(1 - RS/r), 1 / (1 - RS/r)
# ... and so on for all 17 theories.

if __name__ == "__main__":
    start_time = time.time()

    # --- Re-ordered Test Suite (by likelihood of success) ---
    models_to_test = [
        ("Schwarzschild (GR)",         {"func": compression_schwarzschild, "params": {}}),
        ("Acausal (Final State)",      {"func": compression_acausal, "params": {}}),
        # ... (all other 15 models listed in the previously established ranked order) ...
        ("Newtonian Limit",            {"func": compression_newtonian_limit, "params": {}}),
    ]

    print("="*70)
    print("GPU-Accelerated Analysis of Theoretical Models")
    print(f"(Grid Resolution: {NUM_POINTS:,} points)")
    print("="*70)
    print(f"{'Rank':<5} | {'Model Name':<30} | {'Calculated Information Loss':<25}")
    print("-"*70)

    results = []
    # Loop through the re-ordered list of models to test.
    for name, model_info in models_to_test.items():
        # Instantiate the GPU-aware model.
        model = CompressionModel(compression_function=model_info["func"])
        # Calculate loss. All computation happens on the GPU.
        loss = model.calculate_information_loss(r, M, **model_info["params"])
        # --- GPU MODIFICATION: Use .get() to transfer final scalar result from GPU to CPU for storage. ---
        results.append((name, loss.get()))

    # Sort results on the CPU.
    sorted_results = sorted(results, key=lambda item: item[1])

    # Print the final, ranked results table.
    for rank, (name, loss) in enumerate(sorted_results, 1):
        print(f"{rank:<5} | {name:<30} | {loss:.6e}")

    end_time = time.time()
    print("-"*70)
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    print("="*70)
