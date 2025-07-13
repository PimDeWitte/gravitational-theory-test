"""
Example theory implementation.
Copy this file and modify for your own theory.
"""

from base_theory import GravitationalTheory, Tensor
import torch


class ExampleTheory(GravitationalTheory):
    """
    Example gravitational theory implementation.
    
    This is a simple modification of the Schwarzschild metric
    with an additional parameter α that modifies the gravitational strength.
    """
    
    # Specify the category of your theory
    category = "classical"  # Options: "classical", "quantum", "unified"
    
    # Enable caching for efficiency
    cacheable = True
    
    # Optional: Define parameter sweeps for automatic testing
    # sweep = dict(alpha=np.linspace(0.0, 1.0, 5))
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize the theory with parameters.
        
        Args:
            alpha: Modification parameter (0 = pure Schwarzschild)
        """
        super().__init__(f"Example Theory (α={alpha:.2f})")
        self.alpha = torch.as_tensor(alpha, device=torch.device('cpu'), dtype=torch.float32)
    
    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculate the metric components at radius r.
        
        Args:
            r: Radial coordinate
            M_param: Mass of central object
            C_param: Speed of light
            G_param: Gravitational constant
            
        Returns:
            Tuple of (g_tt, g_rr, g_pp, g_tp) metric components
        """
        # Schwarzschild radius
        rs = 2 * G_param * M_param / C_param**2
        
        # Modified metric function
        # This example adds a logarithmic correction term
        m = 1 - rs / r + self.alpha * torch.log(1 + rs / r)
        
        # Ensure m doesn't go to zero
        m = torch.clamp(m, min=1e-10)
        
        # Return metric components
        g_tt = -m  # Time-time component
        g_rr = 1 / m  # Radial-radial component
        g_pp = r**2  # Angular component (spherical symmetry)
        g_tp = torch.zeros_like(r)  # No time-angular coupling (no rotation)
        
        return g_tt, g_rr, g_pp, g_tp
    
    def get_cache_tag(self, N_STEPS: int, precision_tag: str, r0_tag: int) -> str:
        """
        Generate unique cache tag including parameters.
        """
        return f"{self.name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_')}_{N_STEPS}_{precision_tag}_r{r0_tag}" 