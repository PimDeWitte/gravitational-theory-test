# Draft: The Compression Hypothesis

**A computational framework for testing gravitational theories by treating them as information compression algorithms.**

## 

What if gravity isn't just a force, but the universe's way of compressing information?

This project tests that idea by:
1. Treating gravitational theories as "decoders" that reconstruct reality from compressed information
2. Measuring how well each theory reproduces real physics (their "decoding loss")
3. Using orbital mechanics as the test - because orbits are extremely sensitive to spacetime geometry

Think of it like testing video codecs - a good codec reproduces the original perfectly, while a bad one introduces artifacts. Here, General Relativity is our "lossless codec" baseline.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- PyTorch (CPU or GPU)
- 8GB+ RAM recommended

### Theory Directory Structure
All theories are organized in the `theories/` directory:
- `theories/defaults/` - Baseline theories (GR, RN) and standard tests
- `theories/linear_signal_loss/` - Example theory showing unification signals
- `theories/template/` - Template for creating new theories
- macOS/Linux (Windows users: use WSL)

### Installation
```bash
# Clone the repository
git clone https://github.com/pimdewitte/gravity-compression.git
cd gravity-compression

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch numpy matplotlib scipy

# Optional: For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Running Theories

```bash
# Run all default theories with standard validations
./run_theory.sh

# Run a specific theory
./run_theory.sh theories/linear_signal_loss

# Run with high precision
./run_theory.sh theories/linear_signal_loss --final

# Quick test mode
./run_theory.sh --test

# Skip baseline comparisons
./run_theory.sh --skip-defaults theories/my_theory

# Self-discovery mode - AI generates variations
./run_theory.sh theories/linear_signal_loss --self-discover

# Self-discovery with custom prompt
./run_theory.sh theories/einstein_deathbed_unified --self-discover \
  --initial-prompt "explore torsion and asymmetric metrics"

# Run the setup script (creates Python virtual environment)
./setup_gpu.sh

# Test the installation - runs default theories at 1000 steps (~5-10 min)
./scripts/run_validation_tests.sh

# Run the Linear Signal Loss analysis (key discovery)
cd theories/linear_signal_loss
./final_linear_validation_loss.sh
```

## 🔧 Theory Organization

All theories are now organized in a modular structure under `theories/`:

```
theories/
├── defaults/               # Baseline theories and all test theories
│   ├── source/            # Theory implementations
│   ├── grounding/         # Theoretical foundations
│   ├── validations/       # Standard tests
│   └── results/           # Evaluation outputs
├── linear_signal_loss/     # The key discovery
│   ├── source/            # Implementation
│   ├── validations/       # Observation tests
│   └── results/           # Analysis results
└── einstein_deathbed_unified/  # Einstein-inspired theory
    └── source/            # Implementation
```

### Adding Your Own Theory

1. Create a new directory: `theories/your_theory_name/`
2. Add subdirectories: `source/`, `grounding/`, `validations/`, etc.
3. Create `source/theory.py` with your theory class:

```python
from base_theory import GravitationalTheory, Tensor
import torch

class YourTheory(GravitationalTheory):
    category = "classical"  # or "quantum", "unified"
    cacheable = True
    
    def __init__(self):
        super().__init__("Your Theory Name")
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        # Your metric implementation
        g_tt = -(1 - rs/r)  # Example
        g_rr = 1/(1 - rs/r)
        g_pp = r**2
        g_tp = torch.zeros_like(r)  # For torsion/rotation
        return g_tt, g_rr, g_pp, g_tp
```

## 🔬 Running Experiments

### Test Specific Theories
```bash
# Run a specific theory directory
python self_discovery.py --test --theory-dirs theories/linear_signal_loss

# Run multiple theories
python self_discovery.py --test --theory-dirs theories/linear_signal_loss theories/einstein_deathbed_unified

# Run all default theories
python self_discovery.py --test --theory-dirs theories/defaults
```

### High-Precision Validation
```bash
# Run with 5M steps for publication-quality results
python self_discovery.py --final --theory-dirs theories/linear_signal_loss
```

### AI-Assisted Discovery
```bash
# Use AI to generate new theories
export XAI_API_KEY=your_key_here
python self_discovery.py --self-discover --initial-prompt "explore torsion-based unification"
```

## 📊 Dual-Baseline Methodology

We compare against TWO reference theories:
- **Schwarzschild (GR)**: Pure gravity - the "perfect decoder" for mass
- **Reissner-Nordström**: Gravity + electromagnetism - includes charge effects

A unified theory should perform well against BOTH baselines without explicitly including charge.

## 📁 Project Structure

```
gravity_compression/
├── theories/                       # All gravitational theories
│   ├── defaults/                  # Baseline theories and validations
│   │   ├── source/               # GR, RN, and test theories
│   │   ├── validations/          # Standard observational tests
│   │   └── grounding/            # Theoretical foundations
│   ├── linear_signal_loss/        # Example unified theory
│   ├── einstein_deathbed_unified/ # Einstein-inspired theory
│   └── template/                  # Template for new theories
├── self_discovery.py              # Main simulation engine
├── base_theory.py                 # Base class for all theories
├── run_theory.sh                  # Universal theory runner
├── cache/                         # Cached trajectory data
└── papers/                        # Research documentation
```

Each theory directory is self-contained with:
- `source/` - Implementation
- `grounding/` - Theory
- `validations/` - Tests
- `results/` - Outputs
- `self_discovery/` - AI variations
- `runs/` - Simulations

## 🔬 Key Discoveries

### 1. The Linear Signal Loss Finding

The most striking discovery: when we degrade the gravitational signal linearly (like lossy compression), there's a "sweet spot" at γ=0.75 where the theory has equal loss to both pure gravity AND electromagnetism:

```
γ = 0.00 (Pure GR)          γ = 0.75 (Sweet Spot)         γ = 1.00 (Maximum degradation)
      │                              │                               │
      ▼                              ▼                               ▼
Loss vs GR:  0.000                0.153                          0.133
Loss vs RN:  0.250                0.161                          0.133
```

This suggests gravity and electromagnetism might be different "compression settings" of the same underlying process!

### 2. Robustness as a Feature

General Relativity isn't just accurate - it's the most *robust* theory. Like a well-designed codec that handles noise gracefully, GR maintains stability even under extreme conditions where other theories fail.

### 3. Information Loss = Physical Effects

The amount of "compression loss" correlates with observable deviations from GR. This provides a new way to classify and understand alternative theories of gravity.

## 🛠️ Advanced Usage

### Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--test` | Quick test mode (1,000 steps) | False |
| `--final` | High-precision mode (5M steps) | False |
| `--cpu-f64` | Force CPU with float64 precision | False |
| `--self-discover` | Enable AI theory generation | False |
| `--theory-dirs <dirs>` | Theory directories to load | theories/defaults |
| `--initial-prompt <text>` | Seed prompt for AI discovery | "" |

### Example Commands

```bash
# Quick test of Linear Signal Loss
python self_discovery.py --test --theory-dirs theories/linear_signal_loss

# High-precision validation of defaults
python self_discovery.py --final --theory-dirs theories/defaults

# AI discovery with custom prompt
python self_discovery.py --self-discover --initial-prompt "explore quantum corrections to GR"
```

## 📈 Validation Against Observations

The Linear Signal Loss theory can be tested against real astronomical data:

```bash
cd theories/linear_signal_loss/validations
python pulsar_timing_validation.py      # Test against PSR B1913+16
python cassini_ppn_validation.py        # Test PPN parameters
```

## 🤝 Contributing

We welcome contributions! You can:
- Add new theories to `theories/`
- Implement validation tests against observations
- Improve the simulation engine
- Add visualization tools
- Document discoveries

See individual theory READMEs for specific contribution guidelines.

## 📄 Citation

```bibtex
@article{dewitte2025compression,
  title={The Compression Hypothesis: Gravity as Information Compression},
  author={de Witte, Pim},
  journal={arXiv preprint},
  year={2025}
}
```

## 🔬 Creating Your Own Theory

1. **Copy the template**:
   ```bash
   cp -r theories/template theories/my_theory
   ```

2. **Edit `theories/my_theory/source/theory.py`** with your metric:
   ```python
   class MyTheory(GravitationalTheory):
       def get_metric(self, r, M, c, G):
           rs = 2 * G * M / c**2
           # Your metric equations here
           return g_tt, g_rr, g_pp, g_tp
   ```

3. **Run it**:
   ```bash
   ./theories/my_theory/run.sh
   ```

That's it! The framework automatically handles comparisons, validations, and visualizations.

## ⚠️ Important Disclaimers

This project is an exploratory framework for testing gravitational theories through a computational lens. However, please note the following:

- **Potential Circularity**: The simulation setup, including initial conditions and geodesic integration, may introduce circular dependencies on baseline theories like General Relativity. For instance, initial velocity calculations or metric assumptions could bias results towards known models. Users should interpret results cautiously and consider independent validation methods.

- **Need for Rigorous Validation**: While the FFT-based loss metric provides a novel way to compare theories, it is not a substitute for physical experiments or astronomical observations. A more comprehensive test suite, including diverse orbital scenarios, stability analyses, and comparisons with real data (e.g., from LIGO or pulsar timing), is recommended for future development to ensure robustness and reliability.

## 🔗 Links

- [GitHub Repository](https://github.com/pimdewitte/gravity-compression)
- [Research Paper](papers/003/004_not_reviewed)
- [Interactive Visualizations](viz/example_viz.html)

## 📧 Contact

Pim de Witte - pim@generalintuition.ai

---

*"The universe may be the ultimate compression algorithm, and gravity its most elegant implementation."*
