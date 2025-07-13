"""
Pulsar timing validation using PSR B1913+16 periastron advance.
"""

import torch
import numpy as np
from scipy.constants import G, c
from theories.defaults.validations.base_validation import ObservationalValidation
from base_theory import GravitationalTheory


class PulsarTimingValidation(ObservationalValidation):
    """
    Tests theories against the observed periastron advance of PSR B1913+16.
    """
    
    def __init__(self):
        super().__init__(
            name="PSR B1913+16 Periastron Advance",
            description="Tests gravitational theories against binary pulsar orbital precession"
        )
    
    def _load_observational_data(self):
        return {
            'periastron_advance': 4.226595,  # deg/yr (Weisberg & Huang 2016)
            'error': 0.000005,  # deg/yr uncertainty
            'period_days': 0.322997448918,  # Orbital period in days
            'period_seconds': 0.322997448918 * 86400,  # seconds
            'pulsar_mass': 1.438 * 1.989e30,  # kg (1.438 solar masses)
            'companion_mass': 1.390 * 1.989e30,  # kg (1.390 solar masses)
            'eccentricity': 0.6171338,
            'semi_major_axis': 1.95e9,  # meters
        }
    
    def validate(self, theory: GravitationalTheory, **kwargs):
        """
        Run the pulsar timing validation.
        
        kwargs can include:
        - N_STEPS: number of integration steps (default: 50000)
        - device: torch device (default: cpu)
        - dtype: torch dtype (default: float32)
        """
        # Get parameters
        N_STEPS = kwargs.get('N_STEPS', 50000)
        device = kwargs.get('device', torch.device('cpu'))
        dtype = kwargs.get('dtype', torch.float32)
        
        # Set up initial conditions for binary pulsar
        # Using reduced mass approximation
        M_total = self.observational_data['pulsar_mass'] + self.observational_data['companion_mass']
        M_tensor = torch.tensor(M_total, device=device, dtype=dtype)
        
        # Initial radius (periastron)
        a = self.observational_data['semi_major_axis']
        e = self.observational_data['eccentricity']
        r0 = torch.tensor(a * (1 - e), device=device, dtype=dtype)
        
        # Time step
        period = self.observational_data['period_seconds']
        DTau = torch.tensor(period / 1000.0, device=device, dtype=dtype)
        
        # Run simulation
        try:
            # Pass M_tensor as override to use correct mass
            traj = self.run_trajectory_for_validation(
                theory, r0, N_STEPS, DTau, device, dtype, M_override=M_tensor
            )
            
            # Compute periastron advance
            phi = traj[:, 2].cpu().numpy()
            delta_phi = phi[-1] - phi[0]  # Total angle change
            
            # Calculate number of orbits
            total_time = len(traj) * DTau.item()
            num_orbits = total_time / period
            
            if num_orbits == 0:
                return {
                    'pass': False,
                    'observed': self.observational_data['periastron_advance'],
                    'predicted': 0.0,
                    'error': float('inf'),
                    'details': {'error': 'No complete orbits'}
                }
            
            # Advance per orbit
            advance_rad = delta_phi - 2 * np.pi * num_orbits
            advance_deg_per_orbit = np.degrees(advance_rad)
            
            # Convert to deg/yr
            orbits_per_year = 365.25 * 86400 / period
            predicted_advance = advance_deg_per_orbit * orbits_per_year
            
            # Calculate error
            observed = self.observational_data['periastron_advance']
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
                    'advance_per_orbit_deg': advance_deg_per_orbit,
                    'total_simulation_time_days': total_time / 86400
                }
            }
            
        except Exception as e:
            return {
                'pass': False,
                'observed': self.observational_data['periastron_advance'],
                'predicted': float('nan'),
                'error': float('inf'),
                'details': {'error': str(e)}
            }


# Keep legacy function for backward compatibility
def compute_periastron_advance(traj: torch.Tensor, dtau: float) -> float:
    """Legacy function - use PulsarTimingValidation class instead."""
    validator = PulsarTimingValidation()
    period = validator.observational_data['period_seconds']
    phi = traj[:, 2].cpu().numpy()
    delta_phi = phi[-1] - phi[0]
    num_orbits = len(traj) * dtau / period
    if num_orbits == 0: return 0.0
    advance_rad = delta_phi - 2 * np.pi * num_orbits
    advance_deg = np.degrees(float(advance_rad)) / num_orbits * (365.25 * 86400 / period)
    return advance_deg 