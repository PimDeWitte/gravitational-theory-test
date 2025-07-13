# cassini_ppn_validation.py
import torch
import numpy as np
from typing import Dict, Any
from base_validation import ObservationalValidation
from base_theory import GravitationalTheory

class CassiniValidation(ObservationalValidation):
    """Validates theories against Cassini PPN-γ measurements."""
    
    def __init__(self, device=None, dtype=None):
        super().__init__(device, dtype)
        
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """Validate against Cassini PPN-γ parameter observation."""
        
        # Check for verbose mode
        verbose = kwargs.get('verbose', False)
        
        print(f"\n    Starting Cassini PPN-γ validation for {theory.name}...")
        
        # Observed PPN-γ value
        observed_gamma = 1.00000  # ± 2.3e-5
        
        # Calculate predicted γ from theory
        if hasattr(theory, 'get_ppn_gamma'):
            predicted_gamma = theory.get_ppn_gamma()
        else:
            # For all theories, calculate from metric components in weak field
            r_test = self.tensor(1e12, dtype=torch.float64)  # Larger distance to avoid precision loss
            M_sun = self.tensor(1.989e30)  # kg
            
            if verbose:
                print(f"      Computing PPN-γ from metric at r={r_test:.1e} m")
            
            # Get metric components
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_test, M_sun, self.c.item(), self.G.item())
            
            # Compute Newtonian potential (dimensionless)
            Phi = - (self.G * M_sun / (r_test * self.c**2)).to(torch.float64)
            
            # For standard PPN, g_tt ≈ -(1 + 2 Phi), g_rr ≈ 1 + 2 gamma |Phi|
            # So gamma ≈ (g_rr - 1) / (2 |Phi|)
            if abs(Phi) > 0:
                predicted_gamma = ((g_rr.double() - 1) / (2 * torch.abs(Phi))).cpu().item()
            else:
                predicted_gamma = 1.0
            
            # Always print for debugging
            print(f"      Phi = {Phi:.6e}")
            print(f"      g_tt = {g_tt.cpu().item():.6e}")
            print(f"      g_rr = {g_rr.cpu().item():.6e}")
            print(f"      γ = (g_rr - 1) / (2 |Phi|) = {predicted_gamma:.6f}")
        
        error = abs(predicted_gamma - observed_gamma)
        passed = error < 2.3e-5  # Within observed uncertainty
        
        print(f"    Cassini PPN-γ validation complete:")
        print(f"      Observed: {observed_gamma:.6f} ± 2.3e-5")
        print(f"      Predicted: {predicted_gamma:.6f}")
        print(f"      Error: {error:.2e}")
        print(f"      Result: {'PASSED' if passed else 'FAILED'}")
        
        return {
            'test_name': 'Cassini PPN-γ Parameter',
            'observed': observed_gamma,
            'predicted': predicted_gamma,
            'error': error,
            'units': 'dimensionless',
            'passed': bool(passed)  # Convert numpy bool to Python bool
        } 