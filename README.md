# The Compression Hypothesis: Testing Gravity as Information

A computational framework for testing gravitational theories by treating them as information compression algorithms. This project implements the methodology described in "The Compression Hypothesis: Completing Einstein's Final Quest via a Computational Framework for Generating and Testing Gravitational Theories" by Pim de Witte.

## 🚀 Quick Start: Replication Steps

### Prerequisites
- Python 3.8+ 
- PyTorch (CPU or GPU)
- 8GB+ RAM recommended
- macOS (optimized for M3), Linux, or Windows

### Step 1: Setup Environment
```bash
# Clone the repository
git clone https://github.com/pimdewitte/gravity-compression.git
cd gravity-compression

# Run the GPU setup script (creates conda environment and installs dependencies)
./setup_gpu.sh
```

### Step 2: Validate Installation
```bash
# Run validation tests for all theories at 1000 steps
# This tests the computational framework and ensures everything is working
./run_validation_tests.sh
```

This will:
- Test all 69+ gravitational theories
- Generate trajectory plots
- Calculate loss metrics vs Schwarzschild and Reissner-Nordström baselines
- Cache results for faster future runs
- Expected runtime: ~5-10 minutes on GPU, ~30-60 minutes on CPU

### Step 3: Run Linear Signal Loss Analysis
```bash
# Test the unified theory discovery - Linear Signal Loss
# This explores how gravity transforms to electromagnetism as signal degrades
./final_linear_validation_loss.sh
```

This runs the key finding: testing Linear Signal Loss values from γ=0.0 to γ=1.0, demonstrating the potential unification of gravity and electromagnetism through information theory.

### Platform Compatibility
This framework was developed and optimized for **macOS M3** (Apple Silicon) but leverages PyTorch's cross-platform capabilities. It should run on:
- ✅ macOS (Intel/Apple Silicon)
- ✅ Linux (x86/ARM) 
- ✅ Windows (with minor path adjustments)
- ✅ Google Colab / Cloud environments

GPU acceleration is optional - the code automatically falls back to CPU if CUDA/MPS is unavailable.

## 📊 Core Concept: The Compression Hypothesis

### What is the Compression Hypothesis?

The framework tests the radical idea that **gravity is not just curved spacetime, but an emergent information compression process**. Just as a video codec compresses high-dimensional pixel data into a compact representation, gravity may compress the universe's quantum information into the classical spacetime we observe.

### Key Innovation: Dual-Baseline Testing

We evaluate theories against two ground truths:
1. **Schwarzschild metric** (pure gravity) - Tests gravitational fidelity
2. **Reissner-Nordström metric** (gravity + electromagnetism) - Tests unified field potential

### The Discovery: Linear Signal Loss

Our most significant finding is that degrading the gravitational "signal" can produce electromagnetic-like effects:

```python
# Linear Signal Loss transformation
g_tt = -(1 - γ*r_s/r)(1 - r_s/r)

# At γ=0: Pure gravity (Schwarzschild)
# At γ≈0.75-1.0: Equal loss to both baselines → Unification!
```

**Key Result**: At γ ≈ 0.75-1.0, the model achieves nearly equal loss against both gravitational and electromagnetic baselines, suggesting that electromagnetism may emerge from degraded gravitational information.

## 🔬 Methodology Overview

### Three-Tier Validation System

| Mode | Steps | Purpose | Runtime (GPU) |
|------|-------|---------|---------------|
| TEST | 1,000 | Rapid screening | ~0.001s |
| VALIDATION | 100,000 | High-fidelity testing | ~0.1s |
| FINAL | 5,000,000 | Publication quality | ~5s |

### Computational Pipeline

1. **Theory Definition** → Each gravitational theory implements a metric tensor g_μν
2. **Geodesic Integration** → 4th-order Runge-Kutta solver traces particle orbits
3. **FFT Analysis** → Fourier transform extracts orbital frequencies
4. **Loss Computation** → Quantifies deviation from ground truth
5. **Caching** → All results stored for reproducibility

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

## 🧮 Key Theories Tested

### Top Performers (Lowest Loss vs GR)
1. **Schwarzschild (GR)** - 0.000 (baseline)
2. **Newtonian Limit** - 4.63×10⁴ m²
3. **Log Corrected** - 1.45×10⁷ m²
4. **Einstein Final** - 1.58×10⁷ m²

### Promising Unified Candidates
- **Participatory Model** - Incorporates observer effects
- **Linear Signal Loss** - Unifies gravity/EM through information degradation
- **Variable G** - Tests emergent gravitational "constant"

## 🤖 AI-Assisted Theory Discovery

The framework includes an automated discovery loop that:
1. Uses LLMs to interpret Einstein's final notes
2. Generates new metric tensor theories
3. Tests them against both baselines
4. Learns from results to guide future searches

Over 300+ theories have been automatically generated and tested!

## 📈 Visualizing Results

The framework generates polar trajectory plots showing:
- **Red line**: Candidate theory orbit
- **Black dashed**: Schwarzschild (GR) baseline  
- **Blue dotted**: Reissner-Nordström (EM) baseline

Stable orbits indicate valid theories; spiraling/ejection reveals fundamental flaws.

## 🔧 Advanced Usage

### Testing Your Own Theory

```python
# Add to predefined_theories.py or other_generated_theories.py
class MyTheory(GravitationalTheory):
    def get_g_tt(self, r, M, G, C, Rs):
        # Your metric component here
        return -(1 - Rs/r)  # Example: Schwarzschild
```

## 📚 Scientific Background

### Why PyTorch?

Using PyTorch for physics simulations enables:
- **Automatic differentiation** for Christoffel symbols
- **GPU acceleration** without custom CUDA code
- **Cross-platform** compatibility
- **Bridging physics ↔ AI** communities

### Physical Scales

- Test orbit: r = 10 Schwarzschild radii
- Captures relativistic effects (v ≈ 0.22c)
- Detects quantum corrections at 10⁻³⁴ level (validation mode)
- Planck-scale sensitive at 5M steps (final mode)

## 🤝 Contributing

We welcome contributions! Areas of interest:
- New theoretical models
- Performance optimizations  
- Quantum computer integration
- Extended baseline metrics
- Visualization improvements

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{dewitte2025compression,
  title={The Compression Hypothesis},
  author={de Witte, Pim},
  journal={General Intuition PBC},
  year={2025}
}
```

## 🙏 Acknowledgments

This project would not have been possible without the contributions of many individuals:

- **Niko Bonatsos** - Suggestion on using Fourier transforms instead of dot product for comparison, and being my sparring partner throughout this process
- **Google Gemini 2.5 Pro** - Brainstorming on approach and correct formulation of ideas
- **Grok 4** - Making the creation of the automated process 10x easier by rewriting manual logic into a cleanly architected agent with high precision, validating lots of math, and coming up with the approach to fix the RN precision problem!
- **Yann LeCun** - For explaining the V-JEPA architecture so well on a podcast that I was able to draw the parallel to trying to introduce loss to produce the function that ended up correct
- **The PyTorch team** - Exceptional framework that made this research possible

## 📞 Contact

For questions or collaborations:
- GitHub Issues: [github.com/pimdewitte/gravity-compression/issues](https://github.com/pimdewitte/gravity-compression/issues)
- Email: pim@generalintuition.ai

---

*"The universe compresses its quantum complexity into classical simplicity. Gravity is the algorithm."*

## 📊 Key Visual Concepts

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

## 🔍 Understanding the Results

### Loss Metric Interpretation

The "loss" measures how much a theory's predicted orbit deviates from ground truth:

- **Loss = 0**: Perfect match (lossless compression)
- **Loss < 10⁶**: Excellent approximation 
- **Loss > 10⁹**: Fundamental geometric failure

### Fourier Transform Analysis

The FFT decomposes orbits into frequency components:
- **Primary peak**: Orbital period
- **Secondary peaks**: Relativistic precession
- **Missing frequencies**: Incomplete physics

## 🌟 Breakthrough Criteria

A theory qualifies as a breakthrough if it:
1. Achieves lower loss than GR against the R-N baseline
2. Doesn't explicitly include electromagnetic terms
3. Emerges from geometric principles alone
4. Remains stable over millions of integration steps

## 💡 Future Directions

### Quantum Computer Integration
The stochastic noise model suggests quantum computers could:
- Follow particles through quantum foam
- Reconstruct coherent states from noise
- Test superposition of gravitational theories

### Extended Baselines
Future work could add:
- Kerr metric (rotating black holes)
- FLRW metric (cosmological solutions)
- Quantum corrected metrics

### Theory Generation
AI could explore:
- Non-commutative geometries
- Emergent dimensions
- Information-theoretic metrics

## 🎯 Key Takeaways

1. **Gravity = Information Compression**: Physical laws act as codecs
2. **Dual Testing Reveals Unity**: Same framework tests gravity AND electromagnetism  
3. **Signal Loss = Phase Transition**: Degraded gravity becomes electromagnetism
4. **AI Accelerates Discovery**: 300+ theories tested automatically
5. **PyTorch Democratizes Physics**: Anyone can test new theories

---

*Join us in the search for the universe's compression algorithm!*
