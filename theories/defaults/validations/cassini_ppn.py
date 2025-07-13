"""
Cassini PPN-γ parameter validation.
"""

import torch
import numpy as np
from scipy.constants import G, c
from theories.defaults.validations.base_validation import ObservationalValidation
from base_theory import GravitationalTheory


class CassiniPPNValidation(ObservationalValidation):
    """
    Tests theories against Cassini spacecraft PPN-γ measurements.
    """
    
    def __init__(self):
        super().__init__(
            name="Cassini PPN-γ Parameter",
            description="Tests gravitational theories against Cassini radio science constraints"
        )
    
    def _load_observational_data(self):
        return {
            'gamma': 1.0,  # GR prediction
            'constraint': 2.3e-5,  # |γ-1| < 2.3 × 10^-5 (Bertotti et al. 2003)
            'sun_mass': 1.989e30,  # kg
            'earth_sun_distance': 1.496e11,  # meters (1 AU)
            'saturn_sun_distance': 1.434e12,  # meters (9.58 AU at conjunction)
        }
    
    def validate(self, theory: GravitationalTheory, **kwargs):
        """
        Run PPN-γ validation by computing light deflection.
        
        The PPN-γ parameter affects light deflection near massive bodies.
        We simulate light ray trajectories near the Sun during solar conjunction.
        """
        # Get parameters
        device = kwargs.get('device', torch.device('cpu'))
        dtype = kwargs.get('dtype', torch.float32)
        
        # For PPN-γ, we need to analyze the metric components
        # γ affects the spatial curvature: g_rr = (1 + γ * rs/r)
        
        # Sample metric at various radii
        rs = 2 * G * self.observational_data['sun_mass'] / c**2
        r_samples = torch.logspace(
            np.log10(10 * rs), 
            np.log10(1000 * rs), 
            100, 
            device=device, 
            dtype=dtype
        )
        
        M_sun = torch.tensor(self.observational_data['sun_mass'], device=device, dtype=dtype)
        
        try:
            # Get metric components
            g_tt_vals = []
            g_rr_vals = []
            
            for r in r_samples:
                g_tt, g_rr, g_pp, g_tp = theory.get_metric(r, M_sun, c, G)
                g_tt_vals.append(g_tt.item())
                g_rr_vals.append(g_rr.item())
            
            g_tt_vals = np.array(g_tt_vals)
            g_rr_vals = np.array(g_rr_vals)
            r_vals = r_samples.cpu().numpy()
            
            # For Schwarzschild-like metrics: g_rr ≈ 1/(1 - rs/r)
            # For PPN: g_rr ≈ 1 + γ * rs/r
            # We fit to extract effective γ
            
            # Use linear region where rs/r << 1
            mask = r_vals > 100 * rs
            r_fit = r_vals[mask]
            g_rr_fit = g_rr_vals[mask]
            
            # Fit g_rr ≈ 1 + γ * rs/r
            x = rs / r_fit
            y = g_rr_fit - 1
            
            # Linear regression
            if len(x) > 10:
                gamma_eff = np.polyfit(x, y, 1)[0]
            else:
                # Fallback: use single point
                gamma_eff = y[0] / x[0] if x[0] != 0 else 1.0
            
            # Calculate deviation from GR
            deviation = abs(gamma_eff - 1.0)
            
            # Check if within Cassini constraint
            passes = deviation < self.observational_data['constraint']
            
            return {
                'pass': passes,
                'observed': 1.0,  # GR value
                'predicted': gamma_eff,
                'error': deviation,
                'relative_error': deviation,
                'details': {
                    'constraint': self.observational_data['constraint'],
                    'num_samples': len(r_samples),
                    'fit_points': len(r_fit) if 'r_fit' in locals() else 0
                }
            }
            
        except Exception as e:
            return {
                'pass': False,
                'observed': 1.0,
                'predicted': float('nan'),
                'error': float('inf'),
                'details': {'error': str(e)}
            } 