"""
Mercury perihelion precession validation.
"""

import torch
import numpy as np
from scipy.constants import G, c
from theories.defaults.validations.base_validation import ObservationalValidation
from base_theory import GravitationalTheory


class MercuryPerihelionValidation(ObservationalValidation):
    """
    Tests theories against Mercury's observed perihelion precession.
    """
    
    def __init__(self):
        super().__init__(
            name="Mercury Perihelion Precession",
            description="Tests gravitational theories against Mercury's anomalous perihelion advance"
        )
    
    def _load_observational_data(self):
        return {
            'total_precession': 5599.74,  # arcsec/century (total observed)
            'newtonian_precession': 5557.18,  # arcsec/century (from other planets)
            'gr_precession': 42.56,  # arcsec/century (GR contribution)
            'error': 0.41,  # arcsec/century uncertainty
            'semi_major_axis': 5.7909e10,  # meters
            'eccentricity': 0.20563,
            'period_days': 87.969,  # days
            'sun_mass': 1.989e30,  # kg
        }
    
    def validate(self, theory: GravitationalTheory, **kwargs):
        """
        Run Mercury perihelion validation.
        """
        # Get parameters
        N_STEPS = kwargs.get('N_STEPS', 100000)
        device = kwargs.get('device', torch.device('cpu'))
        dtype = kwargs.get('dtype', torch.float32)
        
        # Set up initial conditions
        M_sun = torch.tensor(self.observational_data['sun_mass'], device=device, dtype=dtype)
        
        # Initial radius at perihelion
        a = self.observational_data['semi_major_axis']
        e = self.observational_data['eccentricity']
        r0 = torch.tensor(a * (1 - e), device=device, dtype=dtype)
        
        # Time step (smaller for accuracy)
        period = self.observational_data['period_days'] * 86400  # seconds
        DTau = torch.tensor(period / 5000.0, device=device, dtype=dtype)
        
        # Run simulation
        try:
            # Need to temporarily override M in the validation
            import self_discovery
            old_M = self_discovery.M
            self_discovery.M = M_sun
            
            traj = self.run_trajectory_for_validation(
                theory, r0, N_STEPS, DTau, device, dtype
            )
            
            # Restore original M
            self_discovery.M = old_M
            
            # Find perihelion points (local minima in r)
            r_values = traj[:, 1].cpu().numpy()
            phi_values = traj[:, 2].cpu().numpy()
            
            # Find local minima
            perihelion_indices = []
            for i in range(1, len(r_values) - 1):
                if r_values[i] < r_values[i-1] and r_values[i] < r_values[i+1]:
                    perihelion_indices.append(i)
            
            if len(perihelion_indices) < 2:
                return {
                    'pass': False,
                    'observed': self.observational_data['gr_precession'],
                    'predicted': 0.0,
                    'error': float('inf'),
                    'details': {'error': 'Insufficient perihelion passages'}
                }
            
            # Calculate precession between first and last perihelion
            phi_first = phi_values[perihelion_indices[0]]
            phi_last = phi_values[perihelion_indices[-1]]
            num_orbits = len(perihelion_indices) - 1
            
            # Advance per orbit
            total_advance_rad = (phi_last - phi_first) - 2 * np.pi * num_orbits
            advance_per_orbit_rad = total_advance_rad / num_orbits
            
            # Convert to arcsec/century
            orbits_per_century = 100 * 365.25 / self.observational_data['period_days']
            advance_arcsec_per_orbit = np.degrees(advance_per_orbit_rad) * 3600
            predicted_advance = advance_arcsec_per_orbit * orbits_per_century
            
            # Compare to GR prediction
            observed = self.observational_data['gr_precession']
            error = abs(predicted_advance - observed)
            relative_error = error / observed
            
            # Pass if within 3-sigma
            passes = error < 3 * self.observational_data['error']
            
            return {
                'pass': passes,
                'observed': observed,
                'predicted': predicted_advance,
                'error': error,
                'relative_error': relative_error,
                'details': {
                    'num_orbits': num_orbits,
                    'advance_per_orbit_arcsec': advance_arcsec_per_orbit,
                    'perihelion_passages': len(perihelion_indices)
                }
            }
            
        except Exception as e:
            return {
                'pass': False,
                'observed': self.observational_data['gr_precession'],
                'predicted': float('nan'),
                'error': float('inf'),
                'details': {'error': str(e)}
            } 