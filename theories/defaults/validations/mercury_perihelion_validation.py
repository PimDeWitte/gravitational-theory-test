# mercury_perihelion_validation.py
import torch
import numpy as np
from typing import Dict, Any
from base_validation import ObservationalValidation
from base_theory import GravitationalTheory
from geodesic_integrator import GeodesicIntegrator

class MercuryPerihelionValidation(ObservationalValidation):
    """Validates theories against Mercury perihelion precession."""
    
    def __init__(self, device=None, dtype=None):
        super().__init__(device, dtype)
        
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """Validate against Mercury's observed perihelion advance."""
        
        # Check for verbose mode
        verbose = kwargs.get('verbose', False)
        
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
        
        print(f"\n    Starting Mercury perihelion validation for {theory.name}...")
        if verbose:
            print(f"    Device: {self.device}, Dtype: {self.dtype}")
            print(f"    N_STEPS: {N_STEPS}, DTau: {DTau:.3e}")
            print(f"    Initial r0: {r0:.3e}")
        
        # Get initial conditions
        y0_full = self.get_initial_conditions(theory, r0, M_sun)
        y0_state = y0_full[[0, 1, 2, 4]]
        
        # Initialize integrator
        integrator = GeodesicIntegrator(theory, y0_full, M_sun, self.c, self.G)
        
        # Run simulation
        hist = self.empty((N_STEPS + 1, 4))
        hist[0] = y0_state
        y = y0_state.clone()
        
        perihelion_times = []
        perihelion_angles = []
        
        # Progress logging
        step_print = 100 if kwargs.get('test', False) else 1000
        if verbose:
            step_print = 100
        
        for i in range(N_STEPS):
            y = integrator.rk4_step(y, DTau)
            y = y.to(self.device)  # Explicitly ensure y is on device
            hist[i + 1] = y
            
            # Progress logging
            if (i + 1) % step_print == 0:
                print(f"      Step {i+1}/{N_STEPS} | r={y[1]/a:.3f} AU")
            
            # Improved perihelion detection with interpolation
            if i > 1:
                r_prev2 = hist[i-1, 1]
                r_prev = hist[i, 1]
                r_curr = y[1]
                if r_prev < r_prev2 and r_prev < r_curr:  # Local minimum
                    # Quadratic interpolation for exact min
                    t0, t1, t2 = hist[i-1, 0], hist[i, 0], y[0]
                    r0, r1, r2 = r_prev2, r_prev, r_curr
                    denom = (t0-t1)*(t0-t2)*(t1-t2)
                    A = (t2*(r1-r0) + t1*(r0-r2) + t0*(r2-r1)) / denom if denom != 0 else 0
                    B = ((t2**2)*(r0-r1) + (t1**2)*(r2-r0) + (t0**2)*(r1-r2)) / denom if denom != 0 else 0
                    t_min = -B / (2*A) if A != 0 else t1
                    phi_min = hist[i, 2] + (t_min - t1) * (y[2] - hist[i, 2]) / DTau  # Linear approx
                    perihelion_times.append(t_min)
                    perihelion_angles.append(phi_min)
                    if verbose:
                        print(f"      Perihelion {len(perihelion_times)} at t={t_min:.1f}s, φ={phi_min:.3f} rad")
            
            if not torch.all(torch.isfinite(y)):
                print(f"      Warning: Non-finite values at step {i} - aborting")
                hist = hist[:i+2]
                break
                
        # Ensure entire hist is on device
        hist = hist.to(self.device)
        
        # Calculate precession rate properly
        if len(perihelion_angles) >= 2:
            times_np = np.array([t.cpu().item() for t in perihelion_times])
            angles_np = np.array([phi.cpu().item() for phi in perihelion_angles])
            
            # Fit linear trend to unwrapped angles
            angles_unwrapped = np.unwrap(angles_np)
            coeffs = np.polyfit(times_np, angles_unwrapped, 1)
            if verbose:
                print(f"      Fitted omega_dot = {coeffs[0]:.3e} rad/s")
            
            # Convert to arcseconds per century
            omega_dot = coeffs[0]  # rad/s
            advance_arcsec_century = omega_dot * (100 * 365.25 * 86400) * (206265)  # to arcsec
            
            # Subtract Newtonian (Keplerian) rate
            omega_N = 2 * np.pi / P.cpu().item()
            advance_arcsec_century -= omega_N * (100 * 365.25 * 86400) * (206265)
            
            # For GR reference: expected excess = 3 pi G M / (c^2 a (1-e^2)) per orbit, then per century
            orbits_per_century = (100 * 365.25 * 86400) / P.cpu().item()
            gr_excess_per_orbit = 3 * np.pi * self.G.item() * M_sun.item() / (self.c.item()**2 * a.item() * (1 - e.item()**2))
            gr_advance_century = gr_excess_per_orbit * orbits_per_century * (180 * 3600 / np.pi)  # to arcsec
            if verbose:
                print(f"      GR expected: {gr_advance_century:.2f} arcsec/century")
        else:
            advance_arcsec_century = float('nan')
            if verbose:
                print("      Insufficient perihelions detected")
            
        error = abs(advance_arcsec_century - observed_advance)
        passed = error < 0.1  # Tightened tolerance
        
        print(f"    Mercury perihelion validation complete:")
        print(f"      Observed: {observed_advance:.2f} arcsec/century")
        print(f"      Predicted: {advance_arcsec_century:.2f} arcsec/century")
        print(f"      Error: {error:.2f} arcsec/century")
        print(f"      Result: {'PASSED' if passed else 'FAILED'}")
        
        return {
            'test_name': 'Mercury Perihelion Precession',
            'observed': observed_advance,
            'predicted': advance_arcsec_century,
            'error': error,
            'units': 'arcseconds/century',
            'passed': passed,
            'trajectory': hist.cpu().numpy()
        } 