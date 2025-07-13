# mercury_perihelion_validation.py
import torch
import numpy as np
from typing import Dict, Any
from base_validation import BaseValidation
from base_theory import GravitationalTheory

class MercuryPerihelionValidation(BaseValidation):
    """Validates theories against Mercury perihelion precession."""
    
    def __init__(self, device=None, dtype=None):
        super().__init__(device, dtype)
        
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """Validate against Mercury's observed perihelion advance."""
        
        # Solar mass
        M_sun = self.tensor(1.989e30)  # kg
        
        # Mercury orbital parameters
        a = self.tensor(5.7909e10)  # Semi-major axis (m)
        e = self.tensor(0.2056)  # Eccentricity
        P = self.tensor(87.969 * 86400)  # Orbital period (s)
        
        # Observed excess precession (after accounting for other planets)
        observed_advance = 42.56  # arcseconds per century
        
        # Initial conditions at perihelion
        r_peri = a * (1 - e)
        r0 = r_peri
        
        # Simulation parameters
        n_orbits = 100  # Simulate 100 orbits
        steps_per_orbit = 1000
        N_STEPS = n_orbits * steps_per_orbit
        DTau = P / steps_per_orbit
        
        print(f"    Starting Mercury simulation for {theory.name}...")
        print(f"    Device: {self.device}, Dtype: {self.dtype}")
        
        # Get initial conditions
        y0_full = self.get_initial_conditions(theory, r0, M_sun)
        y0_state = y0_full[[0, 1, 2, 4]]
        
        # Initialize integrator
        from self_discovery import GeodesicIntegrator
        integrator = GeodesicIntegrator(theory, y0_full, M_sun, self.c, self.G)
        
        # Run simulation
        hist = self.empty((N_STEPS + 1, 4))
        hist[0] = y0_state
        y = y0_state.clone()
        
        perihelion_times = []
        perihelion_angles = []
        
        for i in range(N_STEPS):
            y = integrator.rk4_step(y, DTau)
            y = y.to(self.device)  # Explicitly ensure y is on device
            hist[i + 1] = y
            
            # Detect perihelion passages
            if i > 0 and hist[i-1, 1] > hist[i, 1] < y[1]:
                perihelion_times.append(y[0])
                perihelion_angles.append(y[2])
                
            if not torch.all(torch.isfinite(y)):
                break
                
        # New: Ensure entire hist is on device
        hist = hist.to(self.device)
        
        # Calculate precession rate
        if len(perihelion_angles) >= 2:
            times_np = np.array([t.cpu().item() for t in perihelion_times])
            angles_np = np.array([phi.cpu().item() for phi in perihelion_angles])
            
            # Fit advance rate
            angles_unwrapped = np.unwrap(angles_np)
            coeffs = np.polyfit(times_np, angles_unwrapped, 1)
            
            # Convert to arcseconds per century
            omega_dot = coeffs[0]  # rad/s
            advance_arcsec_century = omega_dot * (100 * 365.25 * 86400) * (206265)  # to arcsec
            
            # Subtract Newtonian precession
            omega_N = 2 * np.pi / P.cpu().item()
            advance_arcsec_century -= omega_N * (100 * 365.25 * 86400) * (206265)
            
        else:
            advance_arcsec_century = float('nan')
            
        error = abs(advance_arcsec_century - observed_advance)
        passed = error < 1.0  # Within 1 arcsecond per century
        
        return {
            'test_name': 'Mercury Perihelion Precession',
            'observed': observed_advance,
            'predicted': advance_arcsec_century,
            'error': error,
            'units': 'arcseconds/century',
            'passed': passed,
            'trajectory': hist.cpu().numpy()
        } 