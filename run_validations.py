#!/usr/bin/env python3
import argparse
import os
import sys
import inspect
import json
import torch
from typing import List, Dict, Any
import base_theory  # Ensure base_theory is imported first
from base_validation import ObservationalValidation
import importlib.util
import numpy as np
try:
    import matplotlib
except ImportError:
    matplotlib = None

# Argument parser
def parse_args():
    p = argparse.ArgumentParser(description="Run observational validations on theories")
    p.add_argument("--theory-dirs", type=str, nargs='+', default=["theories/defaults"], help="Theory directories to load")
    p.add_argument("--device", type=str, default="cpu", help="PyTorch device (cpu or mps)")
    p.add_argument("--dtype", type=str, default="float32", help="PyTorch dtype (float32 or float64)")
    p.add_argument("--validate-baselines", action="store_true", help="Also validate baseline theories (normally skipped)")
    p.add_argument("--verbose", action="store_true", help="Enable verbose output during validation")
    p.add_argument("--test", action="store_true", help="Run in test mode with fewer steps")
    return p.parse_args()

def run_all_validations(theories: List, device=None, dtype=None, verbose=False, test=False):
    """
    Run all available validation tests against provided theories.
    
    Args:
        theories: List of theory instances to validate
        device: PyTorch device (if None, will auto-detect)
        dtype: PyTorch dtype (if None, will use float32)
        verbose: Enable verbose output
        test: Run in test mode with fewer steps
    
    Returns:
        Dict with validation results
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
        
    # Load validation modules dynamically
    validators = []
    print("Loading validators from 'theories/defaults/validations/':")
    validator_dir = 'theories/defaults/validations'
    
    if not os.path.exists(validator_dir):
        print(f"ERROR: Validation directory not found: {validator_dir}")
        return {}
        
    files = [f for f in os.listdir(validator_dir) if f.endswith('.py') and not f.startswith('__') and f != 'base_validation.py' and f != 'README.md']
    print(f"  Found files: {files}")
    
    for filename in files:
        module_name = filename[:-3]
        filepath = os.path.join(validator_dir, filename)
        print(f"  Loading {filename} from {filepath}")
        try:
            # Add project root and validator directory to sys.path temporarily
            import sys
            old_path = sys.path.copy()
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            if validator_dir not in sys.path:
                sys.path.insert(0, validator_dir)
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            
            # Inject commonly needed modules into the module's namespace
            module.__dict__['torch'] = torch
            module.__dict__['np'] = np
            module.__dict__['numpy'] = np
            # Ensure matplotlib is available
            try:
                import matplotlib.pyplot as plt
                module.__dict__['plt'] = plt
                module.__dict__['matplotlib'] = matplotlib
            except ImportError:
                print("Warning: matplotlib not available for plotting")
            
            spec.loader.exec_module(module)
            
            # Restore original path
            sys.path = old_path
            
            # Find validation classes
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, ObservationalValidation) and obj != ObservationalValidation:
                    validator = obj(device=device, dtype=dtype)
                    validators.append(validator)
                    print(f"    Loaded: {name}")
        except Exception as e:
            print(f"    Failed to load {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"Initialized {len(validators)} validators")
    
    if not validators:
        print("ERROR: No validators found!")
        return {}
    
    # Run validations
    results = {}
    for theory in theories:
        theory_results = {}
        print(f"\nValidating {theory.name}...")
        
        for validator in validators:
            try:
                result = validator.validate(theory, verbose=verbose, test=test)
                theory_results[result['test_name']] = result
                # Summary already printed by validator
            except Exception as e:
                print(f"  Error running {validator.__class__.__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        results[theory.name] = theory_results
        
    return results

# Function to load theories from a directory
def load_theories_from_dir(theory_dir: str, include_baselines: bool = False) -> List:
    theories = []
    source_dir = os.path.join(theory_dir, 'source')
    print(f'Looking for theories in: {source_dir}')
    if os.path.exists(source_dir):
        theory_files = [f for f in os.listdir(source_dir) if f.endswith('.py') and not f.startswith('__')]
        if not theory_files:
            print(f'  No theory files found in source directory')
        for filename in theory_files:
            filepath = os.path.join(source_dir, filename)
            print(f'  Checking file: {filename}')
            try:
                module_name = f'theory_module_{theory_dir.replace("/", "_")}_{filename[:-3]}'
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, base_theory.GravitationalTheory) and obj != base_theory.GravitationalTheory:
                        print(f'    Found theory: {name}')
                        instance = obj()
                        theories.append(instance)
            except Exception as e:
                print(f'  Failed to load {filename}: {e}')
    else:
        print(f'  No source directory found')
    
    # Optionally load baselines
    if include_baselines:
        baseline_dir = os.path.join(theory_dir, 'baselines')
        if os.path.exists(baseline_dir):
            print(f'  Loading baselines from: {baseline_dir}')
            for filename in os.listdir(baseline_dir):
                if filename.endswith('.py') and not filename.startswith('__'):
                    filepath = os.path.join(baseline_dir, filename)
                    try:
                        module_name = f'baseline_module_{theory_dir.replace("/", "_")}_{filename[:-3]}'
                        spec = importlib.util.spec_from_file_location(module_name, filepath)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and issubclass(obj, base_theory.GravitationalTheory) and obj != base_theory.GravitationalTheory:
                                print(f'    Found baseline: {name}')
                                instance = obj()
                                theories.append(instance)
                    except Exception as e:
                        print(f'  Failed to load baseline {filename}: {e}')

    if theories:
        print(f'  Total theories loaded: {len(theories)}')
    else:
        print('  No theories found')
    return theories

if __name__ == "__main__":
    args = parse_args()
    
    # Set device and dtype
    device = torch.device(args.device if torch.device(args.device).type == 'mps' and torch.backends.mps.is_available() else 'cpu')
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    
    print(f"Running validations on device: {device}, dtype: {dtype}")
    print(f"Theory directories to validate: {args.theory_dirs}")
    print(f"Validate baselines: {args.validate_baselines}")
    
    # Load all theories from all directories
    all_theories = []
    for theory_dir in args.theory_dirs:
        print(f"\nLoading theories from: {theory_dir}")
        theories = load_theories_from_dir(theory_dir, include_baselines=args.validate_baselines)
        all_theories.extend(theories)
    
    print(f"\nTotal theories to validate: {len(all_theories)}")
    
    if not all_theories:
        print("No theories found to validate!")
        sys.exit(1)
    
    # Run validations
    results = run_all_validations(all_theories, device=device, dtype=dtype, verbose=args.verbose, test=args.test)
    
    # Save results
    output_file = 'validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (torch.Tensor, np.number)) else x)
    
    print(f"\nValidation results saved to {output_file}")
    print("\nDone!") 