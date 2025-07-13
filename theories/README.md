# Theories Directory

This directory contains all gravitational theories organized by their specific implementations.

## Structure

Each theory has its own directory with the following subdirectories:

- **source/**: Theory implementation code (theory.py and run.py)
- **baselines/**: Theoretical foundations, derivations, and mathematical background
- **validations/**: Test results against observations and benchmarks
- **papers/**: Related publications and research
- **results/**: Simulation outputs and analysis
- **self_discovery/**: AI-generated variations and explorations
- **runs/**: Individual simulation runs with plots and data

## Special Directories

- **defaults/**: Contains baseline theories (GR, RN) and all standard test theories. These serve as baselines for all other explorations.

## Usage

To run a specific theory:
```bash
python self_discovery.py --theory-dirs theories/linear_signal_loss
```

To run multiple theories:
```bash
python self_discovery.py --theory-dirs theories/linear_signal_loss theories/einstein_deathbed_unified
```

To run all defaults:
```bash
python self_discovery.py --theory-dirs theories/defaults
```

## Adding New Theories

1. Create a new directory: `theories/your_theory_name/`
2. Add subdirectories: `source/`, `baselines/`, etc.
3. Create `source/theory.py` with your GravitationalTheory subclass
4. Optionally add `source/run.py` for standalone testing
5. Document in `baselines/` and add validations 