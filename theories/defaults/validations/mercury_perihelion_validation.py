# mercury_perihelion_validation.py
import torch
import numpy as np
from typing import Dict, Any
from base_validation import ObservationalValidation
from base_theory import GravitationalTheory
from geodesic_integrator import GeodesicIntegrator
import os
import json
from predefined_theories import Schwarzschild  # For GR baseline
import matplotlib.pyplot as plt  # Moved here for always-available
from predefined_theories import ReissnerNordstrom

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
        if kwargs.get('test', False):
            n_orbits = 5  # Reduced for test mode
            steps_per_orbit = 100
        else:
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
        
        gr_perihelion_times = []
        gr_perihelion_angles = []
        rn_perihelion_times = []
        rn_perihelion_angles = []
        
        # Progress logging
        step_print = 100 if kwargs.get('test', False) else 1000
        if verbose:
            step_print = 100
        
        gr_theory = Schwarzschild()
        gr_y0_full = self.get_initial_conditions(gr_theory, r0, M_sun)
        gr_y0_state = gr_y0_full[[0, 1, 2, 4]]
        gr_integrator = GeodesicIntegrator(gr_theory, gr_y0_full, M_sun, self.c, self.G)
        gr_hist = self.empty((N_STEPS + 1, 4))
        gr_hist[0] = gr_y0_state
        gr_y = gr_y0_state.clone()
        
        rn_theory = ReissnerNordstrom(Q=0.0)
        rn_y0_full = self.get_initial_conditions(rn_theory, r0, M_sun)
        rn_y0_state = rn_y0_full[[0, 1, 2, 4]]
        rn_integrator = GeodesicIntegrator(rn_theory, rn_y0_full, M_sun, self.c, self.G)
        rn_hist = self.empty((N_STEPS + 1, 4))
        rn_hist[0] = rn_y0_state
        rn_y = rn_y0_state.clone()
        
        for i in range(N_STEPS):
            y = integrator.rk4_step(y, DTau)
            y = y.to(self.device)  # Explicitly ensure y is on device
            hist[i + 1] = y
            
            gr_y = gr_integrator.rk4_step(gr_y, DTau)
            gr_hist[i + 1] = gr_y
            
            rn_y = rn_integrator.rk4_step(rn_y, DTau)
            rn_hist[i + 1] = rn_y
            
            # Progress logging
            if (i + 1) % step_print == 0:
                print(f"      Step {i+1}/{N_STEPS} | predicted r={y[1]/a:.3f} AU | expected (GR) r={gr_y[1]/a:.3f} AU | RN r={rn_y[1]/a:.3f} AU")
            
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
            
            if i > 1:
                gr_r_prev2 = gr_hist[i-1, 1]
                gr_r_prev = gr_hist[i, 1]
                gr_r_curr = gr_y[1]
                if gr_r_prev < gr_r_prev2 and gr_r_prev < gr_r_curr:
                    t0, t1, t2 = gr_hist[i-1, 0], gr_hist[i, 0], gr_y[0]
                    r0, r1, r2 = gr_r_prev2, gr_r_prev, gr_r_curr
                    denom = (t0-t1)*(t0-t2)*(t1-t2)
                    A = (t2*(r1-r0) + t1*(r0-r2) + t0*(r2-r1)) / denom if denom != 0 else 0
                    B = ((t2**2)*(r0-r1) + (t1**2)*(r2-r0) + (t0**2)*(r1-r2)) / denom if denom != 0 else 0
                    t_min = -B / (2*A) if A != 0 else t1
                    phi_min = gr_hist[i, 2] + (t_min - t1) * (gr_y[2] - gr_hist[i, 2]) / DTau
                    gr_perihelion_times.append(t_min)
                    gr_perihelion_angles.append(phi_min)
                    if verbose:
                        print(f"      GR Perihelion {len(gr_perihelion_times)} at t={t_min:.1f}s, φ={phi_min:.3f} rad")
            
            if i > 1:
                rn_r_prev2 = rn_hist[i-1, 1]
                rn_r_prev = rn_hist[i, 1]
                rn_r_curr = rn_y[1]
                if rn_r_prev < rn_r_prev2 and rn_r_prev < rn_r_curr:
                    t0, t1, t2 = rn_hist[i-1, 0], rn_hist[i, 0], rn_y[0]
                    r0, r1, r2 = rn_r_prev2, rn_r_prev, rn_r_curr
                    denom = (t0-t1)*(t0-t2)*(t1-t2)
                    A = (t2*(r1-r0) + t1*(r0-r2) + t0*(r2-r1)) / denom if denom != 0 else 0
                    B = ((t2**2)*(r0-r1) + (t1**2)*(r2-r0) + (t0**2)*(r1-r2)) / denom if denom != 0 else 0
                    t_min = -B / (2*A) if A != 0 else t1
                    phi_min = rn_hist[i, 2] + (t_min - t1) * (rn_y[2] - rn_hist[i, 2]) / DTau
                    rn_perihelion_times.append(t_min)
                    rn_perihelion_angles.append(phi_min)
                    if verbose:
                        print(f"      RN Perihelion {len(rn_perihelion_times)} at t={t_min:.1f}s, φ={phi_min:.3f} rad")
            
            if not torch.all(torch.isfinite(y)):
                print(f"      Warning: Non-finite values at step {i} - aborting")
                hist = hist[:i+2]
                break
                
        # Ensure entire hist is on device
        hist = hist.to(self.device)
        
        # New: Run GR baseline for comparison
        gr_hist = gr_hist.to(self.device)
        
        # Compute running MSE vs. GR
        mse_vs_gr = torch.cumsum((hist[:, 1] - gr_hist[:, 1])**2, dim=0) / (torch.arange(1, len(hist)+1, device=self.device, dtype=self.dtype))
        if verbose:
            print(f"      Final MSE vs. GR: {mse_vs_gr[-1]:.3e}")
        
        # Log progress every 10%
        if verbose:
            for pct in range(10, 101, 10):
                step = int(N_STEPS * pct / 100)
                current_mse = mse_vs_gr[step-1]
                prev_mse = mse_vs_gr[step//2] if step > 1 else 0
                trend = 'improving' if current_mse < prev_mse else 'degrading'
                print(f"      At {pct}% (step {step}): MSE={current_mse:.3e} ({trend})")
        
        # Export trajectory to JSON for visualizer
        is_test = kwargs.get('test', False)
        downsample_factor = 10 if is_test else 1  # Downsample in test mode
        traj_sim = hist[::downsample_factor, [1,2]].cpu().tolist()
        traj_gr = gr_hist[::downsample_factor, [1,2]].cpu().tolist()
        traj_data = {
            'sim': traj_sim,
            'gr': traj_gr,
            'theory': theory.name
        }
        json_path = f'{os.path.dirname(__file__)}/mercury_traj_{theory.name.replace(" ", "_").replace("(", "").replace(")", "")}.json'
        with open(json_path, 'w') as f:
            json.dump(traj_data, f)
        
        # Generate viz.html with auto-load and scaling
        viz_html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Mercury Trajectory Viz: {theory.name}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <!-- Add your visualization_enhanced.js -->
    <script src="../../../viz/visualization_enhanced.js"></script>
</head>
<body>
    <div id="viz"></div>
    <script>
        fetch('{os.path.basename(json_path)}')
            .then(res => res.json())
            .then(data => {{
                const visualizer = new GravityVisualizerEnhanced('viz');  // Assume it has a loadTrajectory method
                visualizer.loadTrajectory(data.sim, data.gr);
                visualizer.autoScale();  // Implement in JS: compute bounds and set camera
            }})
            .catch(err => console.error('Failed to load trajectory:', err));
    </script>
</body>
</html>
'''
        viz_path = f'{os.path.dirname(__file__)}/mercury_viz_{theory.name.replace(" ", "_").replace("(", "").replace(")", "")}.html'
        with open(viz_path, 'w') as f:
            f.write(viz_html)
        print(f"      Generated visualizer: {viz_path}")

        # New: Compute running accuracy vs. approximate Keplerian (Newtonian) baseline
        # For simple elliptical orbit: r(θ) = a (1 - e^2) / (1 + e cos θ)
        kepler_r = a * (1 - e**2) / (1 + e * torch.cos(hist[:, 2]))
        running_mse = torch.cumsum((hist[:, 1] - kepler_r)**2, dim=0) / (torch.arange(1, len(hist)+1, device=self.device, dtype=self.dtype))
        if verbose:
            print(f"      Final running MSE vs. Kepler: {running_mse[-1]:.3e}")
        
        # Plot r vs. step to visualize stability
        try:
            is_test = kwargs.get('test', False)
            downsample = 10 if is_test else 1
            steps = range(0, len(hist), downsample)
            r_sim = hist[::downsample, 1].cpu().numpy()
            r_kepler = kepler_r[::downsample].cpu().numpy()
            plt.figure(figsize=(10, 6))
            plt.plot(steps, r_sim, label='Simulated r')
            plt.plot(steps, r_kepler, '--', label='Kepler Approx')
            plt.xlabel('Step')
            plt.ylabel('r (m)')
            plt.title(f'r vs. Step for {theory.name}')
            plt.legend()
            plt.savefig(f'{os.path.dirname(__file__)}/mercury_r_plot_{theory.name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
            plt.close()
        except Exception as e:
            print(f"      Warning: Failed to generate r plot: {e}")

        # Polar trajectory plot
        try:
            is_test = kwargs.get('test', False)
            downsample = 10 if is_test else 1
            phi = hist[::downsample, 2].cpu().numpy()
            r_norm = hist[::downsample, 1].cpu().numpy() / 1.496e11  # Normalize to AU
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection='polar')
            ax.plot(phi, r_norm, label='Simulated')
            ax.set_title(f'Trajectory for {theory.name}')
            plt.savefig(f'{os.path.dirname(__file__)}/mercury_trajectory_{theory.name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
            plt.close()
        except Exception as e:
            print(f"      Warning: Failed to generate trajectory plot: {e}")
        
        # Generate interactive viz
        self.generate_viz(theory, hist, os.path.dirname(__file__), gr_hist=gr_hist)

        # Calculate precession rate properly
        def compute_advance(perihelion_times, perihelion_angles, P, verbose):
            if len(perihelion_angles) >= 2:
                times_np = np.array([t.cpu().item() for t in perihelion_times])
                angles_np = np.array([phi.cpu().item() for phi in perihelion_angles])
                angles_unwrapped = np.unwrap(angles_np)
                coeffs = np.polyfit(times_np, angles_unwrapped, 1)
                if verbose:
                    print(f"      Fitted omega_dot = {coeffs[0]:.3e} rad/s")
                omega_dot = coeffs[0]
                advance_arcsec_century = omega_dot * (100 * 365.25 * 86400) * (206265)
                omega_N = 2 * np.pi / P.cpu().item()
                advance_arcsec_century -= omega_N * (100 * 365.25 * 86400) * (206265)
                return advance_arcsec_century
            else:
                return float('nan')

        advance_theory = compute_advance(perihelion_times, perihelion_angles, P, verbose)
        advance_gr = compute_advance(gr_perihelion_times, gr_perihelion_angles, P, verbose)
        advance_rn = compute_advance(rn_perihelion_times, rn_perihelion_angles, P, verbose)
        
        print(f"    Mercury perihelion validation complete:")
        print(f"      Observed: {observed_advance:.2f} arcsec/century")
        print(f"      Predicted (Theory): {advance_theory:.2f} arcsec/century")
        print(f"      Predicted (GR): {advance_gr:.2f} arcsec/century")
        print(f"      Predicted (RN Q=0): {advance_rn:.2f} arcsec/century")
        error = abs(advance_theory - observed_advance)
        passed = error < 0.1  # Tightened tolerance
        
        print(f"      Error: {error:.2f} arcsec/century")
        print(f"      Result: {'PASSED' if passed else 'FAILED'}")
        
        return {
            'test_name': 'Mercury Perihelion Precession',
            'observed': observed_advance,
            'predicted': advance_theory,
            'predicted_gr': advance_gr,
            'predicted_rn': advance_rn,
            'error': error,
            'units': 'arcseconds/century',
            'passed': bool(passed),  # Convert numpy bool to Python bool
            'trajectory': hist.cpu().numpy()
        } 