# Logging System Updates

## Overview
The logging system has been updated to be minimal by default, with verbose output available via the `--verbose` flag.

## Changes Made

### 1. Added `--verbose` Flag
- Added to `self_discovery.py` argument parser
- Can be passed via `run_theory.sh` using `--verbose`
- Triggers detailed logging for debugging

### 2. Warning System
- If `--verbose` is used without `--test`, a warning is displayed
- The warning explains that verbose logging significantly impacts performance
- 3-second pause to ensure user sees the warning

### 3. Default Logging (Non-Verbose)
Shows only essential information:
- Theory names being evaluated
- Cache loading/generation messages
- Progress updates every 10% (10 total updates per simulation)
- Critical errors and simulation aborts
- Final loss calculations
- Significant torsion detection (g_tp > 1e-6)

### 4. Verbose Logging
When `--verbose` is enabled, additional output includes:
- API prompts and responses
- Generated theory code (raw and cleaned)
- Step-by-step device and finiteness checks
- Detailed optimization iterations
- All torsion detections (even tiny values)
- Initial conditions calculations
- Integrator parameters (E, Lz)
- Non-finite value details
- More frequent progress updates (50 total)

### 5. Performance Considerations
- Verbose mode should primarily be used with `--test` mode (1000 steps)
- Running verbose with full simulations (100K+ steps) will significantly slow performance
- The extra logging calls and string formatting add overhead

## Usage Examples

### Standard Run (Minimal Logging)
```bash
./run_theory.sh theories/linear_signal_loss
```

### Test Run with Verbose Debugging
```bash
./run_theory.sh theories/linear_signal_loss --test --verbose
```

### Validation with Verbose Output
```bash
./run_theory.sh theories/defaults --validate-baselines --test --verbose
```

## Implementation Details

The verbose flag is checked throughout the codebase:
- `self_discovery.py`: Main simulation and AI generation
- `base_validation.py`: Validation framework
- `theories/defaults/validations/base_validation.py`: Theory validation

The flag is accessed via `args.verbose` where `args` is imported from `self_discovery`. 