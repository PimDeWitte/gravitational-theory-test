import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add all necessary imports for the validation framework
import torch

# Import validation classes
exec(open('theories/defaults/validations/base_validation.py').read())
exec(open('theories/defaults/validations/pulsar_timing.py').read())
exec(open('theories/defaults/validations/mercury_perihelion.py').read())
exec(open('theories/defaults/validations/cassini_ppn.py').read())

# Import theory loading function
from self_discovery import load_theories_from_dirs

# Get theory directories from command line
theory_dirs = sys.argv[1].split()

# Load theories
all_theories = {}
for theory_dir in theory_dirs:
    theories = load_theories_from_dirs([theory_dir])
    all_theories.update(theories)

# Initialize validators
validators = [
    PulsarTimingValidation(),
    MercuryPerihelionValidation(),
    CassiniPPNValidation(),
]

# Run validations
for theory_dir, theories in all_theories.items():
    print(f"\nValidating theories from: {theory_dir}")
    print("-" * 50)
    
    for theory in theories:
        print(f"\nTheory: {theory.name}")
        
        for validator in validators:
            result = validator.validate(theory)
            status = "PASS" if result['pass'] else "FAIL"
            print(f"  {validator.name}: {status}")
            print(f"    Observed: {result['observed']:.6f}")
            print(f"    Predicted: {result['predicted']:.6f}")
            print(f"    Error: {result['error']:.6f}")
            
            if 'details' in result and 'error' in result['details']:
                print(f"    Error: {result['details']['error']}")
