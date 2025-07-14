#!/usr/bin/env python3
"""
Quick test script to run validations with reduced step counts.
Use this instead of full validations when debugging.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'theories', 'defaults', 'baselines'))

from run_validations import run_all_validations
from schwarzschild import Schwarzschild
from reissner_nordstrom import ReissnerNordstrom
import torch

# Quick test with reduced steps
device = torch.device("cpu")  # Use CPU for testing
dtype = torch.float32

# Just test with two theories
theories = [
    Schwarzschild(),
    ReissnerNordstrom(Q=1e19)
]

print("Running validations in TEST MODE (reduced steps)...")
print("=" * 60)

results = run_all_validations(
    theories=theories,
    device=device,
    dtype=dtype,
    verbose=True,
    test=True  # This enables reduced step counts
)

print("\n" + "=" * 60)
print("SUMMARY:")
for theory_name, theory_results in results.items():
    print(f"\n{theory_name}:")
    for test_name, result in theory_results.items():
        status = "PASSED" if result.get('passed', False) else "FAILED"
        print(f"  {test_name}: {status}") 