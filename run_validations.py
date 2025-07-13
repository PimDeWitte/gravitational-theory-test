# run_validations.py
import torch
from typing import List, Dict, Any

def run_all_validations(theories: List, device=None, dtype=None):
    """
    Run all validations ensuring consistent device/dtype usage.
    
    Args:
        theories: List of theory instances to validate
        device: PyTorch device (if None, auto-detect like self_discovery.py)
        dtype: PyTorch dtype (if None, use float32 or inherit from globals)
    """
    # Set up device/dtype (same logic as self_discovery.py)
    if device is None:
        try:
            from self_discovery import device as global_device
            device = global_device
        except ImportError:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if dtype is None:
        try:
            from self_discovery import DTYPE as global_dtype
            dtype = global_dtype
        except ImportError:
            dtype = torch.float32
    
    print(f"\nRunning validations on device: {device}, dtype: {dtype}")
    
    # Import validation modules from validations/ subdirectory
    from validations.pulsar_timing_validation import PulsarTimingValidation
    from validations.mercury_perihelion_validation import MercuryPerihelionValidation
    from validations.cassini_ppn_validation import CassiniValidation
    
    # Initialize validators with correct device/dtype
    validators = [
        PulsarTimingValidation(device=device, dtype=dtype),
        MercuryPerihelionValidation(device=device, dtype=dtype),
        CassiniValidation(device=device, dtype=dtype)
    ]
    
    results = {}
    
    for theory in theories:
        theory_results = {}
        print(f"\nValidating: {theory.name}")
        
        for validator in validators:
            try:
                result = validator.validate(theory)
                theory_results[result['test_name']] = result
                
                status = "PASS" if result['passed'] else "FAIL"
                print(f"  {result['test_name']}: {status}")
                print(f"    Observed: {result['observed']:.6f} {result['units']}")
                print(f"    Predicted: {result['predicted']:.6f} {result['units']}")
                print(f"    Error: {result['error']:.6f} {result['units']}")
                
            except Exception as e:
                print(f"  {validator.__class__.__name__}: ERROR - {str(e)}")
                theory_results[validator.__class__.__name__] = {
                    'error': str(e),
                    'passed': False
                }
        
        results[theory.name] = theory_results
    
    return results 