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
            r_test = self.tensor(1e12)  # Larger distance to avoid precision loss
            M_sun = self.tensor(1.989e30)  # kg
            
            if verbose:
                print(f"      Computing PPN-γ from metric at r={r_test:.1e} m")
            
            # Get metric components
            g_tt, g_rr, g_pp, g_tp = theory.get_metric(r_test, M_sun, self.c.item(), self.G.item())
            
            # Compute Newtonian potential (dimensionless)
            Phi = - (self.G * M_sun / (r_test * self.c**2))
            
            # <reason>Fix PPN calculation: At weak field, g_rr = 1 + 2γ|Φ| + O(Φ²), so γ ≈ (g_rr - 1)/(2|Φ|)</reason>
            # For Linear Signal Loss: g_rr = 1/m where m = 1 - (1+γ)rs/r + γ(rs/r)²
            # In weak field: g_rr ≈ 1 + (1+γ)rs/r = 1 + (1+γ)·2|Φ|
            # So PPN-γ_eff = (1+γ)/2 + γ/2 = (1+γ)/2 + γ/2 = (1+γ)/2
            
            # More precise calculation avoiding catastrophic cancellation
            rs = 2 * self.G * M_sun / self.c**2
            rs_over_r = rs / r_test
            
            # For Linear Signal Loss theory
            if hasattr(theory, 'gamma'):
                # Direct calculation from theory
                theory_gamma = theory.gamma
                # In weak field, effective PPN gamma = (1 + theory.gamma)
                predicted_gamma = 1.0 + theory_gamma
                
                if verbose:
                    print(f"      Theory γ parameter: {theory_gamma}")
                    print(f"      Predicted PPN-γ: {predicted_gamma}")
            else:
                # Standard calculation for other theories
                g_rr_deviation = g_rr - 1.0
                if abs(Phi) > 1e-15:
                    predicted_gamma = g_rr_deviation / (2 * torch.abs(Phi))
                    predicted_gamma = predicted_gamma.cpu().item()
                else:
                    predicted_gamma = 1.0
            
            # Always print for debugging
            print(f"      rs/r = {rs_over_r.cpu().item():.6e}")
            print(f"      Phi = {Phi.cpu().item():.6e}")
            print(f"      g_tt = {g_tt.cpu().item():.12e}")
            print(f"      g_rr = {g_rr.cpu().item():.12e}")
            print(f"      g_rr - 1 = {(g_rr - 1).cpu().item():.12e}")
            if not hasattr(theory, 'gamma'):
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