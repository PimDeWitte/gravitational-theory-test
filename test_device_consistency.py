#!/usr/bin/env python3
"""
Test script to verify device consistency across the gravity compression framework.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_theory import GravitationalTheory
from theories.defaults.baselines.schwarzschild import Schwarzschild
from theories.defaults.baselines.reissner_nordstrom import ReissnerNordstrom
from theories.linear_signal_loss.source.theory import LinearSignalLoss
from theories.einstein_deathbed_unified.source.theory import EinsteinDeathbedUnified


def test_theory_device_consistency(theory: GravitationalTheory, device: torch.device, dtype: torch.dtype):
    """Test that a theory handles device placement correctly."""
    print(f"\nTesting {theory.name} on device {device} with dtype {dtype}")
    
    # Create test inputs on specified device
    r = torch.tensor(10.0, device=device, dtype=dtype)
    M = torch.tensor(1e30, device=device, dtype=dtype)
    c = 3e8
    G = 6.67e-11
    
    try:
        # Get metric components
        g_tt, g_rr, g_pp, g_tp = theory.get_metric(r, M, c, G)
        
        # Check all outputs are on correct device
        components = {'g_tt': g_tt, 'g_rr': g_rr, 'g_pp': g_pp, 'g_tp': g_tp}
        for name, comp in components.items():
            if comp.device != device:
                print(f"  ❌ FAIL: {name} is on {comp.device}, expected {device}")
                return False
            if comp.dtype != dtype:
                print(f"  ❌ FAIL: {name} has dtype {comp.dtype}, expected {dtype}")
                return False
        
        print(f"  ✅ PASS: All metric components on correct device and dtype")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception raised: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_geodesic_integrator_device_consistency(device: torch.device, dtype: torch.dtype):
    """Test that GeodesicIntegrator handles device placement correctly."""
    print(f"\nTesting GeodesicIntegrator on device {device} with dtype {dtype}")
    
    try:
        from self_discovery import GeodesicIntegrator
        
        # Create test theory and initial conditions
        theory = Schwarzschild()
        
        # Initial state on specified device
        y0_full = torch.tensor([0.0, 10.0, 0.0, 1.0, 0.0, 0.1], device=device, dtype=dtype)
        M = torch.tensor(1e30, device=device, dtype=dtype)
        c = 3e8
        G = 6.67e-11
        
        # Create integrator
        integrator = GeodesicIntegrator(theory, y0_full, M, c, G)
        
        # Test RK4 step
        y_state = y0_full[[0, 1, 2, 4]]
        y_next = integrator.rk4_step(y_state, 0.1)
        
        if y_next.device != device:
            print(f"  ❌ FAIL: RK4 output is on {y_next.device}, expected {device}")
            return False
        if y_next.dtype != dtype:
            print(f"  ❌ FAIL: RK4 output has dtype {y_next.dtype}, expected {dtype}")
            return False
            
        print(f"  ✅ PASS: GeodesicIntegrator maintains device consistency")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Exception raised: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all device consistency tests."""
    print("="*60)
    print("Device Consistency Tests for Gravity Compression Framework")
    print("="*60)
    
    # Detect available device
    if torch.cuda.is_available():
        devices = [torch.device('cpu'), torch.device('cuda:0')]
    elif torch.backends.mps.is_available():
        devices = [torch.device('cpu'), torch.device('mps')]
    else:
        devices = [torch.device('cpu')]
    
    dtypes = [torch.float32, torch.float64]
    
    # Test theories
    theories = [
        Schwarzschild(),
        ReissnerNordstrom(Q=1e19),
        LinearSignalLoss(gamma=0.5),
        EinsteinDeathbedUnified(alpha=1/137)
    ]
    
    all_passed = True
    
    # Test each theory on each device/dtype combination
    for device in devices:
        for dtype in dtypes:
            print(f"\n{'='*60}")
            print(f"Testing on device: {device}, dtype: {dtype}")
            print(f"{'='*60}")
            
            # Test theories
            for theory in theories:
                if not test_theory_device_consistency(theory, device, dtype):
                    all_passed = False
            
            # Test GeodesicIntegrator
            if not test_geodesic_integrator_device_consistency(device, dtype):
                all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print(f"{'='*60}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main()) 