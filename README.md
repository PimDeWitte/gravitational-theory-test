# Feedback Draft: The Compression Hypothesis: Testing Gravity as Information

A computational framework for testing gravitational theories by treating them as information compression algorithms.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- PyTorch (CPU or GPU)
- 8GB+ RAM recommended

### Setup
```bash
# Clone the repository
git clone https://github.com/pimdewitte/gravity-compression.git
cd gravity-compression

# Run the GPU setup script (creates conda environment and installs dependencies)
./setup_gpu.sh

# Validate installation - tests all theories at 1000 steps (~5-10 min)
./run_validation_tests.sh

# Run Linear Signal Loss analysis (key finding)
./final_linear_validation_loss.sh
```

## 🔧 Adding Your Own Theory

Add to `predefined_theories.py` or `other_generated_theories.py`:

```python
class MyTheory(GravitationalTheory):
    def get_g_tt(self, r, M, G, C, Rs):
        # Your metric component here
        return -(1 - Rs/r)  # Example: Schwarzschild
```

Then test it:
```bash
python test_gravity_theory.py --theory MyTheory --num_steps 1000
```

## 📁 Project Structure

```
gravity_compression/
├── setup_gpu.sh              # Environment setup script
├── run_validation_tests.sh   # Validation test runner
├── final_linear_validation_loss.sh  # Linear signal loss analysis
├── test_gravity_theory.py    # Main simulation engine
├── predefined_theories.py    # 69+ gravitational theories
├── linear_signal_loss.py     # Signal degradation model
├── cache/                    # Cached trajectory data
├── runs/                     # Simulation results & plots
└── papers/                   # Documentation & papers
```

## 📄 Citation

```bibtex
@article{dewitte2025compression,
  title={The Compression Hypothesis},
  author={de Witte, Pim},
  journal={General Intuition PBC},
  year={2025}
}
```

## 📊 Visual Overview

### The Compression Hypothesis Illustrated

```
┌─────────────────────────────────────────────────────────────┐
│                  The Compression Hypothesis                  │
│                                                              │
│  Quantum State          GRAVITY           Classical Spacetime│
│  (High-dimensional) ──────────────────▶  (4D Observable)    │
│       |                                          |           │
│       |               Theory g_μν                |           │
│       |                Decoder                   |           │
│       └──────────────────┬──────────────────────┘           │
│                          │                                   │
│                    Decoding Loss                             │
│               (How well does the theory                      │
│                reconstruct reality?)                         │
└─────────────────────────────────────────────────────────────┘
```

### Computational Pipeline Flow

```
┌──────────────────┐
│ Gravitational    │
│ Theory g_μν      │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Metric Tensor    │ ← PyTorch tensors handle 4D spacetime
│ Components       │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Christoffel      │ ← Automatic differentiation via autograd
│ Symbols Γ        │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Geodesic         │ ← 4th-order Runge-Kutta integration
│ Integration      │
└────────┬─────────┘
         │
┌────────▼─────────┐
│ Trajectory r(t)  │ ← Orbital path in spacetime
└────────┬─────────┘
         │
┌────────▼─────────┐
│ FFT Analysis     │ ← Extract orbital frequencies
└────────┬─────────┘
         │
┌────────▼─────────┐     ┌─────────────────┐
│ Compare with     │────▶│ Schwarzschild   │
│ Dual Baselines   │     │ (Pure Gravity)  │
│                  │     └─────────────────┘
│                  │     ┌─────────────────┐
│                  │────▶│ Reissner-       │
│                  │     │ Nordström (EM)  │
└────────┬─────────┘     └─────────────────┘
         │
┌────────▼─────────┐
│ Loss Computation │ → Quantitative ranking
└──────────────────┘
```

### The Linear Signal Loss Discovery

The most profound result shows how gravity transforms into electromagnetism through signal degradation:

```
γ = 0.00 (Pure GR)          γ = 0.75 (Critical Point)         γ = 1.00 (EM-like)
      │                              │                               │
      ▼                              ▼                               ▼
┌─────────────┐            ┌─────────────┐                ┌─────────────┐
│ Loss vs GR: │            │ Loss vs GR: │                │ Loss vs GR: │
│   0.000     │            │   0.153     │                │   0.133     │
├─────────────┤            ├─────────────┤                ├─────────────┤
│ Loss vs RN: │            │ Loss vs RN: │                │ Loss vs RN: │
│   0.269     │            │   0.161     │                │   0.133     │
└─────────────┘            └─────────────┘                └─────────────┘
      │                              │                               │
      └──────────────────────────────┴───────────────────────────────┘
                                     │
                              UNIFIED REGIME
                         (Equal loss to both!)
```

### Why This Matters: A New Paradigm

Traditional Approach:
```
Einstein Field Equations → Curved Spacetime → Gravitational Effects
```

Compression Hypothesis:
```
Quantum Information → Compression Algorithm → Classical Spacetime
                           │
                           └─→ Gravity emerges from optimization
```
