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
- macOS/Linux (Windows users: use WSL)

### Installation
```bash
# Clone the repository
git clone https://github.com/pimdewitte/gravity-compression.git
cd gravity-compression

# Run the setup script (creates Python virtual environment)
./setup_gpu.sh

# Test the installation - runs all theories at 1000 steps (~5-10 min)
./run_validation_tests.sh

# Run the Linear Signal Loss analysis (key discovery)
./final_linear_validation_loss.sh
```

## 🔧 Adding Your Own Theory

Create a new theory by adding to `predefined_theories.py` or creating a new file:

```python
class MyTheory(GravitationalTheory):
    def __init__(self):
        super().__init__("My Custom Theory")
    
    def get_metric(self, r, M_param, C_param, G_param):
        # Calculate the 4 metric components
        rs = 2 * G_param * M_param / C_param**2  # Schwarzschild radius
        
        # Example: Standard Schwarzschild metric
        g_tt = -(1 - rs/r)      # Time component
        g_rr = 1/(1 - rs/r)     # Radial component  
        g_phiphi = r**2         # Angular component
        g_tphi = 0              # Time-angular coupling (0 for symmetric)
        
        return g_tt, g_rr, g_phiphi, torch.zeros_like(r)
```

Then test it:
```bash
# For theories in predefined_theories.py (auto-loaded)
python self_discovery.py --test

# For theories in a separate file
python self_discovery.py --test --manual-theories-file my_theories.py
```

## 🏗️ How It Works

### The Core Concept

```
High-Dimensional          GRAVITY              Low-Dimensional
Quantum State      ═══════════════════▶      Classical Spacetime
(Hidden reality)         (Decoder)            (What we observe)
      │                                              │
      │              Theory = g_μν                   │
      └──────────── Decoding Loss ──────────────────┘
                 "How well does it work?"
```

### The Pipeline

1. **Define a Theory**: Each theory specifies how spacetime curves (the metric tensor)
2. **Calculate Motion**: Use the metric to compute how objects move (geodesics)
3. **Simulate Orbits**: Integrate the equations of motion for ~100,000 steps
4. **Extract Frequencies**: Use FFT to analyze the orbital frequency spectrum
5. **Compare to Baselines**: Measure deviation from General Relativity and Reissner-Nordström
6. **Rank by Loss**: Lower loss = better theory (better "decoder")

### Key Insight: Dual Baselines

We compare against TWO reference theories:
- **Schwarzschild (GR)**: Pure gravity - the "perfect decoder" for mass
- **Reissner-Nordström**: Gravity + electromagnetism - includes charge effects

A unified theory should perform well against BOTH baselines without explicitly including charge.

## 📁 Project Structure

```
gravity_compression/
├── setup_gpu.sh                    # Environment setup script
├── run_validation_tests.sh         # Quick validation runner
├── final_linear_validation_loss.sh # Linear signal loss experiment
├── self_discovery.py               # Main simulation engine
├── predefined_theories.py          # Core baseline theories (Schwarzschild, Reissner-Nordström)
├── other_generated_theories.py     # Collection of test theories (40+ variants)
├── linear_signal_loss.py           # Signal degradation experiment
├── cache/                          # Cached trajectory data (speeds up repeated runs)
├── runs/                           # Results, plots, and analysis
├── generated_codes/                # AI-generated theory attempts
└── papers/                         # Research papers and documentation
```

**Note**: Many theories include parameter sweeps. For example, `LinearSignalLoss` tests 5 different γ values, 
so the total number of theory instances tested can exceed 100+ when all sweeps are included.

## 🔬 Key Discoveries

### 1. The Linear Signal Loss Finding

The most striking discovery: when we degrade the gravitational signal linearly (like lossy compression), there's a "sweet spot" at γ=0.75 where the theory has equal loss to both pure gravity AND electromagnetism:

```
γ = 0.00 (Pure GR)          γ = 0.75 (Sweet Spot)         γ = 1.00 (Maximum degradation)
      │                              │                               │
      ▼                              ▼                               ▼
Loss vs GR:  0.000              Loss vs GR:  0.153              Loss vs GR:  0.133
Loss vs RN:  0.269              Loss vs RN:  0.161              Loss vs RN:  0.133
                                     │
                              UNIFIED REGIME
                         (Equal distance to both!)
```

This suggests that controlled information loss might be key to unification.

### 2. AI-Assisted Theory Discovery

The framework includes an AI loop that:
- Learns from previous results
- Generates new theories inspired by Einstein's work
- Tests them automatically
- Searches for breakthrough unifications

Run with: `python self_discovery.py --self-discover`

## 📊 Understanding the Results

When you run a test, you'll see rankings like:

```
--- RANKING vs. GENERAL RELATIVITY (GR) ---
Rank | Model                          | Loss (FFT MSE)
-----|--------------------------------|---------------
   1 | Schwarzschild (GR)             | 0.000e+00  ← Perfect (it IS the baseline)
   2 | Einstein Regularised Core      | 2.451e-06  ← Tiny deviation
   3 | Log Corrected (β=+0.17)        | 1.823e-04  ← Small quantum correction
   4 | Variable G (δ=-0.10)           | 3.421e-03  ← Noticeable difference
  ...
```

Lower loss = closer to the baseline = better information preservation.

### Generated Visualizations

For each theory tested, the framework generates:

1. **Orbital Trajectory Plot** (`plot.png`): Shows the actual orbit in polar coordinates
   - Black dashed line: Schwarzschild (pure gravity) baseline
   - Blue dotted line: Reissner-Nordström (gravity + charge) baseline
   - Red solid line: The theory being tested
   - Green dot: Starting position
   - Red X: Ending position

2. **Metric Components Plot** (`metric_plot.png`): Shows the 4 metric tensor components
   - g_tt: Time dilation effects
   - g_rr: Radial space curvature
   - g_φφ: Angular space scaling
   - g_tφ: Time-space coupling (0 for symmetric theories)

All results are saved in timestamped directories under `runs/`.

## ⚙️ Command-Line Options

### Basic Usage
```bash
python self_discovery.py [options]
```

### Available Options

| Flag | Description | Default |
|------|-------------|---------|
| `--test` | Quick test mode with 1,000 steps | 100,000 steps |
| `--final` | High-precision mode with 5M steps and float64 | float32, 100k steps |
| `--cpu-f64` | Force CPU with float64 precision | GPU with float32 |
| `--self-discover` | Enable AI theory generation loop | Disabled |
| `--initial-prompt "..."` | Seed prompt for AI generation | Empty |
| `--api-provider [grok/gemini/openai/anthropic]` | Choose AI provider | grok |
| `--manual-theories-file <file.py>` | Load custom theories from file | None |

### Example Commands

```bash
# Quick test of all predefined theories
python self_discovery.py --test

# Test your custom theories
python self_discovery.py --test --manual-theories-file my_theories.py

# High-precision validation
python self_discovery.py --final

# AI-assisted theory discovery
export XAI_API_KEY=your_key_here
python self_discovery.py --self-discover

# AI discovery with custom prompt
python self_discovery.py --self-discover --initial-prompt "explore torsion-based unification"

# Run specific experiment
python self_discovery.py --final --manual-theories-file linear_signal_loss.py
```
## 📄 Citation

```bibtex
@article{dewitte2025compression,
  title={The Compression Hypothesis: Gravity as Information Compression},
  author={de Witte, Pim},
  journal={arXiv preprint},
  year={2025}
}
```

## 🚧 Future Work

- Test more sophisticated "lossy compression" theories
- Explore theories that interpolate between GR and RN
- Add quantum noise and measure stability

---
