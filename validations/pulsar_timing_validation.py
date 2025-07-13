# pulsar_timing_validation.py
import torch
import numpy as np
from typing import Dict, Any
from base_validation import BaseValidation
from base_theory import GravitationalTheory

class PulsarTimingValidation(BaseValidation):
    """Validates theories against pulsar timing observations."""
    
    def __init__(self, device=None, dtype=None):
        super().__init__(device, dtype)
        
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """
        Validate a theory against PSR B1913+16 observations.
        """
        # System parameters (all as tensors on correct device)
        M1 = self.tensor(1.4408 * 1.989e30)  # Primary mass (kg)
        M2 = self.tensor(1.3886 * 1.989e30)  # Secondary mass (kg)
        M_total = M1 + M2
        
        # Orbital parameters
        a = self.tensor(1.95e9)  # Semi-major axis (m)
        e = self.tensor(0.6171338)  # Eccentricity
        P = self.tensor(27906.98163)  # Orbital period (s)
        
        # Observed periastron advance
        observed_advance = 4.226595  # degrees per year
        
        # Initial conditions at periastron
        r_peri = a * (1 - e)
        r0 = r_peri
        
        # Simulation parameters
        n_orbits = 10
        steps_per_orbit = 5000
        N_STEPS = n_orbits * steps_per_orbit
        DTau = P / steps_per_orbit
        
        print(f"    Starting trajectory simulation for {theory.name}...")
        print(f"    Device: {self.device}, Dtype: {self.dtype}")
        print(f"    N_STEPS: {N_STEPS}, DTau: {DTau:.3e}")
        print(f"    Initial r0: {r0:.3e}")
        
        # Get initial conditions (using same method as self_discovery.py)
        y0_full = self.get_initial_conditions(theory, r0, M_total)
        y0_state = y0_full[[0, 1, 2, 4]]  # [t, r, phi, dr/dtau]
        
        # Initialize integrator (simplified version)
        from self_discovery import GeodesicIntegrator
        integrator = GeodesicIntegrator(theory, y0_full, M_total, self.c, self.G)
        
        # Trajectory storage
        hist = self.empty((N_STEPS + 1, 4))
        hist[0] = y0_state
        
        # Run simulation
        y = y0_state.clone()
        periastron_times = []
        periastron_angles = []
        last_r = r0
        
        for i in range(N_STEPS):
            # RK4 step
            y = integrator.rk4_step(y, DTau)
            y = y.to(self.device)  # Explicitly ensure y is on device
            hist[i + 1] = y
            
            # Check for periastron passage (local minimum in r)
            current_r = y[1]
            if i > 0 and last_r > hist[i, 1] < current_r:
                # Interpolate to find exact periastron
                t_peri = y[0]
                phi_peri = y[2]
                periastron_times.append(t_peri)
                periastron_angles.append(phi_peri)
            
            last_r = hist[i, 1]
            
            # Safety check
            if not torch.all(torch.isfinite(y)):
                print(f"    Warning: Non-finite values at step {i}")
                break
                
        # New: Ensure entire hist is on device
        hist = hist.to(self.device)
        
        # Calculate periastron advance
        if len(periastron_angles) >= 2:
            # Convert to numpy for polyfit (ensure on CPU)
            times_np = np.array([t.cpu().item() for t in periastron_times])
            angles_np = np.array([phi.cpu().item() for phi in periastron_angles])
            
            # Fit linear trend to unwrapped angles
            angles_unwrapped = np.unwrap(angles_np)
            coeffs = np.polyfit(times_np, angles_unwrapped, 1)
            
            # Calculate advance rate
            omega_dot = coeffs[0]  # rad/s
            advance_deg_yr = omega_dot * (365.25 * 86400) * (180 / np.pi)
            
            # Subtract Newtonian contribution
            omega_N = 2 * np.pi / P.cpu().item()
            advance_deg_yr -= omega_N * (365.25 * 86400) * (180 / np.pi)
            
        else:
            advance_deg_yr = float('nan')
            
        # Calculate error
        error = abs(advance_deg_yr - observed_advance)
        passed = error < 0.1  # Within 0.1 deg/yr
        
        return {
            'test_name': 'PSR B1913+16 Periastron Advance',
            'observed': observed_advance,
            'predicted': advance_deg_yr,
            'error': error,
            'units': 'degrees/year',
            'passed': passed,
            'trajectory': hist.cpu().numpy()
        } 