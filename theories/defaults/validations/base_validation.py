"""
Base validation framework for testing gravitational theories against observations.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from base_theory import GravitationalTheory
import time


class ObservationalValidation(ABC):
    """
    Abstract base class for observational validations.
    Each validation test should inherit from this and implement the validate method.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.observational_data = self._load_observational_data()
    
    @abstractmethod
    def _load_observational_data(self) -> Dict[str, Any]:
        """Load the real observational data for this test."""
        pass
    
    @abstractmethod
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """
        Run the validation test on a theory.
        
        Returns:
            Dict containing at minimum:
            - 'pass': bool - whether the theory passes this test
            - 'observed': float - the observed value
            - 'predicted': float - the theory's prediction
            - 'error': float - absolute or relative error
            - 'details': dict - additional test-specific information
        """
        pass
    
    def run_trajectory_for_validation(self, theory: GravitationalTheory, 
                                    r0: torch.Tensor, N_STEPS: int, 
                                    DTau: float, device: torch.device, 
                                    dtype: torch.dtype, M_override: torch.Tensor = None) -> torch.Tensor:
        """
        Helper method to run a trajectory simulation for validation purposes with caching.
        """
        from self_discovery import GeodesicIntegrator, get_initial_conditions, M, c, G, EPSILON, RS
        import os
        import json
        import time
        
        # Adjust N_STEPS for pulsar (high-step test)
        if 'PSR' in self.name:
            N_STEPS = min(N_STEPS, 10000)  # Reduce for performance
        
        # Create cache tag
        precision_tag = "f64" if dtype == torch.float64 else "f32"
        r0_tag = int(r0.item() / RS.item()) if 'RS' in globals() else int(r0.item())
        mass_tag = f"M{int(M_override.item()) if M_override is not None else 'default'}"
        tag = f"{self.name}_{theory.name}_{N_STEPS}_{precision_tag}_{r0_tag}_{mass_tag}".replace(' ', '_').replace('(', '').replace(')', '')
        cache_dir = 'cache/validations'
        os.makedirs(cache_dir, exist_ok=True)
        fname = f"{cache_dir}/cache_{tag}.pt"
        
        if os.path.exists(fname):
            print(f"    Loading cached trajectory: {tag}")
            return torch.load(fname, map_location=device)
        
        print(f"    Generating new trajectory: {tag}")
        
        print(f"    Starting trajectory simulation for {self.name}...")
        print(f"    N_STEPS: {N_STEPS}, DTau: {DTau.item() if hasattr(DTau, 'item') else DTau:.3e}")
        print(f"    Initial r0: {r0.item() if hasattr(r0, 'item') else r0:.3e}")
        
        # Use override mass if provided
        M_use = M_override if M_override is not None else M
        
        # For validation, we need to compute initial conditions with the correct mass
        # This is a simplified version that doesn't use the problematic optimization
        if M_override is not None:
            print(f"    Using override mass: {M_use.item():.3e} kg")
            # Simple circular orbit initial conditions
            v_tan = torch.sqrt(G * M_use / r0)
            g_tt0, _, g_pp0, g_tp0 = theory.get_metric(r0, M_use, c, G)
            
            # Approximate initial velocities for circular orbit
            norm_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * c)) ** 2
            dt_dtau0 = 1.0 / torch.sqrt(norm_sq + 1e-10)
            dphi_dtau0 = (v_tan / r0) * dt_dtau0
            
            y0_full = torch.tensor([0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()], 
                                 device=device, dtype=dtype)
            print(f"    Using simplified initial conditions for validation")
        else:
            # Get initial conditions using the standard method
            print("    Getting initial conditions...")
            y0_full = get_initial_conditions(theory, r0)
        
        y0_state = y0_full[[0, 1, 2, 4]].clone()
        
        print(f"    Initial state: t={y0_state[0].item():.3e}, r={y0_state[1].item():.3e}, phi={y0_state[2].item():.3e}, dr/dtau={y0_state[3].item():.3e}")
        
        # Create integrator with correct mass
        integ = GeodesicIntegrator(theory, y0_full, M_use, c.item() if hasattr(c, 'item') else c, G.item() if hasattr(G, 'item') else G)
        print(f"    Integrator E={integ.E.item():.3e}, Lz={integ.Lz.item():.3e}")
        
        # Run simulation
        hist = torch.empty((N_STEPS + 1, 4), device=device, dtype=dtype)
        hist[0] = y0_state
        y = y0_state.clone()
        
        print(f"    Running simulation...")
        step_print = max(1, N_STEPS // 10)  # Print 10 progress updates
        start_time = time.time()
        
        for i in range(N_STEPS):
            y = integ.rk4_step(y, DTau)
            hist[i + 1] = y
            
            # Progress logging
            if (i + 1) % step_print == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (N_STEPS - i - 1) / rate
                print(f"      Step {i+1}/{N_STEPS} ({100*(i+1)/N_STEPS:.1f}%) | r={y[1].item():.3e} | {rate:.0f} steps/s | ETA: {eta:.1f}s")
            
            # Check for failures
            if not torch.all(torch.isfinite(y)):
                print(f"    WARNING: Non-finite values at step {i+1}, aborting simulation")
                print(f"    State: {y}")
                hist = hist[:i+2]
                break
                
            # Check if fallen into black hole
            if y[1] <= r0 * 0.1:  # Using r0 as proxy for RS
                print(f"    Particle reached horizon at step {i+1}")
                hist = hist[:i+2]
                break
            
            # Add timeout check
            if time.time() - start_time > 600:  # 10 minute timeout
                print(f"    WARNING: Simulation timeout after {i+1} steps")
                hist = hist[:i+2]
                break
            
            # In the loop, add rate check after 1000 steps
            if i == 999:  # After 1000 steps
                elapsed_1000 = time.time() - start_time
                rate_1000 = 1000 / elapsed_1000
                if rate_1000 < 100:
                    print(f"    WARNING: Low rate ({rate_1000:.1f} steps/s) - early exit")
                    hist = hist[:i+1]
                    break
        
        total_time = time.time() - start_time
        print(f"    Simulation completed in {total_time:.1f}s ({len(hist)-1} steps)")
        
        # After simulation, save to cache
        torch.save(hist, fname)
        debug_dict = {
            "test_name": self.name,
            "theory": theory.name,
            "tag": tag,
            "N_STEPS": len(hist)-1,
            "DTau": DTau.item(),
            "r0": r0.item(),
            "device": str(device),
            "dtype": str(dtype),
            "timestamp": time.strftime("%Y%m%d_%H%M%S")
        }
        with open(f"{fname}.json", "w") as f:
            json.dump(debug_dict, f, indent=4)
        
        return hist 