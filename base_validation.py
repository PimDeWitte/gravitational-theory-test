#!/usr/bin/env python3
"""
Base validation module with proper device and dtype handling.
Ensures all tensors are created on the same device as the main simulation.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from scipy.constants import G, c, epsilon_0, hbar
import math

# Import base theory if available
try:
    from base_theory import GravitationalTheory
except ImportError:
    # Fallback definition if base_theory not in path
    class GravitationalTheory:
        pass

class BaseValidation:
    """
    Base class for validation tests that properly handles device and dtype.
    Ensures all tensors are created on the same device as the main simulation.
    """
    
    def __init__(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """
        Initialize validation with proper device and dtype handling.
        
        Args:
            device: PyTorch device. If None, will try to inherit from globals or auto-detect.
            dtype: PyTorch dtype. If None, will try to inherit from globals or use float32.
        """
        # Try to inherit from main script globals first
        if device is None:
            if 'device' in globals():
                self.device = globals()['device']
            else:
                # Auto-detect device (same logic as self_discovery.py)
                self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        if dtype is None:
            if 'DTYPE' in globals():
                self.dtype = globals()['DTYPE']
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype
            
        # Also get other constants if available
        self.EPSILON = globals().get('EPSILON', torch.finfo(self.dtype).eps * 100)
        
        # Physical constants as tensors on correct device
        self.G = self.tensor(G)
        self.c = self.tensor(c)
        self.epsilon_0 = self.tensor(epsilon_0)
        self.hbar = self.tensor(hbar)
        
    def tensor(self, data, **kwargs):
        """Convert data to tensor with correct device and dtype."""
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=self.device, dtype=self.dtype)
            # Check if verbose mode is enabled
            try:
                from self_discovery import args
                if args.verbose and data.device != self.device:
                    print(f"Debug: Moving tensor from {data.device} to {self.device}")
            except:
                pass
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(device=self.device, dtype=self.dtype)
            try:
                from self_discovery import args
                if args.verbose:
                    print(f"Debug: Created tensor from numpy on {tensor.device}")
            except:
                pass
        else:
            tensor = torch.tensor(data, device=self.device, dtype=self.dtype, **kwargs)
            try:
                from self_discovery import args
                if args.verbose:
                    print(f"Debug: Created tensor from scalar/list on {tensor.device}")
            except:
                pass
        return tensor
    
    def zeros(self, *shape) -> torch.Tensor:
        """Create zeros tensor on correct device."""
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)
    
    def ones(self, *shape) -> torch.Tensor:
        """Create ones tensor on correct device."""
        return torch.ones(*shape, device=self.device, dtype=self.dtype)
    
    def linspace(self, start: float, end: float, steps: int) -> torch.Tensor:
        """Create linspace tensor on correct device."""
        return torch.linspace(start, end, steps, device=self.device, dtype=self.dtype)
    
    def arange(self, *args) -> torch.Tensor:
        """Create arange tensor on correct device."""
        return torch.arange(*args, device=self.device, dtype=self.dtype)
    
    def empty(self, *shape) -> torch.Tensor:
        """Create empty tensor on correct device."""
        return torch.empty(*shape, device=self.device, dtype=self.dtype)
    
    def stack(self, tensors: list) -> torch.Tensor:
        """Stack tensors, ensuring all are on correct device."""
        return torch.stack([self.tensor(t) for t in tensors])
    
    def cat(self, tensors: list, dim: int = 0) -> torch.Tensor:
        """Concatenate tensors, ensuring all are on correct device."""
        return torch.cat([self.tensor(t) for t in tensors], dim=dim)
    
    def load(self, path: str) -> torch.Tensor:
        """Load tensor and map to correct device."""
        return torch.load(path, map_location=self.device)
    
    def get_initial_conditions(self, model: GravitationalTheory, r0: torch.Tensor, 
                             M: torch.Tensor) -> torch.Tensor:
        """
        Compute initial conditions (same as self_discovery.py).
        Returns full state vector [t, r, phi, dt/dtau, dr/dtau, dphi/dtau].
        """
        # Approximate circular orbit velocity
        v_tan = torch.sqrt(self.G * M / r0)
        
        # Get metric at initial position
        g_tt0, _, g_pp0, g_tp0 = model.get_metric(r0, M, self.c.item(), self.G.item())
        
        # Normalize 4-velocity
        norm_sq = -g_tt0 - g_pp0 * (v_tan / (r0 * self.c)) ** 2
        dt_dtau0 = 1.0 / torch.sqrt(norm_sq + self.EPSILON)
        dphi_dtau0 = (v_tan / r0) * dt_dtau0
        
        # Return full state vector
        return self.tensor([0.0, r0.item(), 0.0, dt_dtau0.item(), 0.0, dphi_dtau0.item()])


class ObservationalValidation(BaseValidation):
    """
    Base class for observational validation tests.
    All validation implementations should inherit from this class.
    """
    
    def validate(self, theory: GravitationalTheory, **kwargs) -> Dict[str, Any]:
        """
        Validate a theory against observational data.
        
        Args:
            theory: The gravitational theory to validate
            **kwargs: Additional parameters for validation
            
        Returns:
            Dict containing:
                - test_name: Name of the validation test
                - observed: Observed value
                - predicted: Predicted value
                - error: Absolute or relative error
                - units: Units of measurement
                - passed: Boolean indicating if test passed
                - trajectory: Optional trajectory data
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    def generate_viz(self, theory: GravitationalTheory, hist: torch.Tensor, output_dir: str, gr_hist: Optional[torch.Tensor] = None):
        """
        Generate interactive 3D visualization of trajectory using Three.js.
        Saves an HTML file with embedded JS for rendering orbits.
        """
        import json
        import os
        
        # Downsample for performance
        downsample = 1 if len(hist) < 10000 else len(hist) // 5000  # Max ~5000 points
        traj_sim = hist[::downsample, [1,2]].cpu().tolist()  # r, phi to [x,y] in JS
        traj_gr = gr_hist[::downsample, [1,2]].cpu().tolist() if gr_hist is not None else []
        
        data_path = os.path.join(output_dir, f'traj_{theory.name.replace(" ", "_").replace("(", "").replace(")", "")}.json')
        with open(data_path, 'w') as f:
            json.dump({'sim': traj_sim, 'gr': traj_gr, 'theory': theory.name}, f)
        
        viz_path = os.path.join(output_dir, f'viz_{theory.name.replace(" ", "_").replace("(", "").replace(")", "")}.html')
        html = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Orbit Viz: {theory.name}</title>
    <style> body {{ margin: 0; }} </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <script>
        fetch('{os.path.basename(data_path)}').then(res => res.json()).then(data => {{
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer();
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            
            // Function to create orbit line
            function createOrbit(points, color) {{
                const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0] * Math.cos(p[1]), p[0] * Math.sin(p[1]), 0)));
                const material = new THREE.LineBasicMaterial({{color}});
                return new THREE.Line(geometry, material);
            }}
            
            scene.add(createOrbit(data.sim, 0xff0000));  // Red for sim
            if (data.gr.length) scene.add(createOrbit(data.gr, 0x00ff00));  // Green for GR
            
            // Auto-scale camera
            const allPoints = data.sim.concat(data.gr);
            const maxR = Math.max(...allPoints.map(p => p[0])) * 1.2;
            camera.position.z = maxR;
            
            function animate() {{
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }}
            animate();
        }});
    </script>
</body>
</html>
'''
        with open(viz_path, 'w') as f:
            f.write(html)
        project_root = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.relpath(os.path.abspath(viz_path), project_root)
        print(f'      Interactive visualization available at: http://localhost:8000/{relative_path}')
        return viz_path


# Example usage (for standalone testing)
if __name__ == "__main__":
    val = BaseValidation()
    print(f"Device: {val.device}")
    print(f"DType: {val.dtype}")
    test_tensor = val.tensor([1.0, 2.0])
    print(f"Test tensor device: {test_tensor.device}") 