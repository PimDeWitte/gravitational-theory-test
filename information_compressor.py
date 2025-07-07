import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import G, c, k_b, hbar

# --- STEP 1: SETUP AND GROUND TRUTH ---
# We define a central mass M. For simplicity, we'll use one solar mass.
M_SOLAR = 1.989e30
M = M_SOLAR

# Calculate the "ground truth" Schwarzschild Radius for this mass.
SCHWARZSCHILD_RADIUS = (2 * G * M) / (c**2)

# Create a grid of points 'r' for our space.
r = np.linspace(SCHWARZSCHILD_RADIUS * 1.1, SCHWARZSCHILD_RADIUS * 5, 500)

def get_schwarzschild_solution(r_grid, rs):
    """Returns the correct g_tt and g_rr components for a given mass."""
    g_tt = -(1 - rs / r_grid)
    g_rr = 1 / (1 - rs / r_grid)
    return g_tt, g_rr

# --- STEP 2: THE INFORMATION COMPRESSION MODEL ---

class CompressionModel:
    """
    Tests different 'compression algorithms' to see if they produce the
    correct spacetime geometry by minimizing information loss.
    """
    def __init__(self, compression_function):
        """
        Args:
            compression_function (callable): A function that takes (r, M) and
                                             proposes a spacetime metric (g_tt, g_rr).
                                             This is your "compression algorithm".
        """
        self.compression_function = compression_function

    def get_source_information(self, mass):
        """
        Postulate: The fundamental information content of a source is
        proportional to its mass-energy. E=mc^2.
        Let's assume a simple linear relationship for this model.
        """
        # This is a key theoretical assumption you can also modify.
        # We add a constant scaling factor to match units later.
        info_scaling_factor = (np.pi * k_b * c**3) / (G * hbar)
        return info_scaling_factor * mass**2

    def get_geometric_information_capacity(self, g_rr_at_horizon):
        """
        Calculates the information capacity (entropy) of a given geometry.
        This capacity depends on the area of the event horizon.
        """
        # The radius of the horizon in a given geometry is where g_rr -> infinity.
        # We find this by looking for the max value in our proposed g_rr.
        if np.max(g_rr_at_horizon) < 1e5: # No horizon found
            return 0
        
        horizon_index = np.argmax(g_rr_at_horizon)
        rs_predicted = r[horizon_index]
        
        # Area of the predicted event horizon
        area = 4 * np.pi * rs_predicted**2
        
        # Bekenstein-Hawking formula: S = (k_b * c^3 * A) / (4 * G * hbar)
        entropy = (k_b * c**3 * area) / (4 * G * hbar)
        return entropy

    def calculate_information_loss(self, r_grid, mass):
        """
        The core of the model. It calculates how well the proposed geometry
        (the compressed file) represents the source information.
        Loss = (Capacity of Geometry - Information of Source)^2
        """
        # Generate the proposed metric using your compression algorithm
        g_tt_proposed, g_rr_proposed = self.compression_function(r_grid, mass)
        
        # Calculate the two information values
        source_info = self.get_source_information(mass)
        geom_info_capacity = self.get_geometric_information_capacity(g_rr_proposed)
        
        # The loss is the squared difference. We want this to be zero.
        loss = (geom_info_capacity - source_info)**2
        return loss, g_tt_proposed, g_rr_proposed

# --- STEP 3: EXAMPLE INPUTS (YOUR COMPRESSION ALGORITHMS) ---

def compression_flat_space(r, M):
    """Hypothesis: No compression occurs. Spacetime is flat."""
    g_tt = np.full_like(r, -1.0)
    g_rr = np.full_like(r, 1.0)
    return g_tt, g_rr

def compression_linear_falloff(r, M):
    """Hypothesis: Compression effect falls off linearly with distance."""
    # This is an intuitive but incorrect guess.
    rs = (2 * G * M) / (c**2)
    effect = rs / r
    g_tt = -1 + effect
    g_rr = 1 + effect # Incorrect relationship
    return g_tt, g_rr

def compression_perfect_encoding(r, M):
    """Hypothesis: The correct compression algorithm from General Relativity."""
    # This function should yield zero information loss.
    rs = (2 * G * M) / (c**2)
    g_tt = -(1 - rs / r)
    g_rr = 1 / (1 - rs / r)
    return g_tt, g_rr

# --- STEP 4: RUN THE SIMULATION ---
if __name__ == "__main__":
    # --- CHOOSE YOUR COMPRESSION ALGORITHM HERE ---
    # current_algorithm = compression_flat_space
    # current_algorithm = compression_linear_falloff
    current_algorithm = compression_perfect_encoding
    # --------------------------------------------

    print(f"Testing Algorithm: '{current_algorithm.__name__}'")
    model = CompressionModel(compression_function=current_algorithm)
    
    # Run the model to get the loss and the predicted geometry
    loss, g_tt_pred, g_rr_pred = model.calculate_information_loss(r, M)
    
    print(f"\n▶ Information Loss: {loss:.4e}")
    print("  (A perfect algorithm has a loss of 0.0)")

    # Get the correct answer for comparison
    g_tt_target, g_rr_target = get_schwarzschild_solution(r, SCHWARZSCHILD_RADIUS)

    # Plot the results
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Algorithm: {current_algorithm.__name__}", fontsize=16)

    ax1.plot(r / SCHWARZSCHILD_RADIUS, g_tt_target, 'c-', label='Target Geometry', linewidth=4, alpha=0.8)
    ax1.plot(r / SCHWARZSCHILD_RADIUS, g_tt_pred, 'm--', label='Your Compressed Geometry', linewidth=2)
    ax1.set_title('Time Dilation Component (g_tt)')
    ax1.set_ylabel('Metric Value')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    ax2.plot(r / SCHWARZSCHILD_RADIUS, g_rr_target, 'c-', label='Target Geometry', linewidth=4, alpha=0.8)
    ax2.plot(r / SCHWARZSCHILD_RADIUS, g_rr_pred, 'm--', label='Your Compressed Geometry', linewidth=2)
    ax2.set_title('Space Curvature Component (g_rr)')
    ax2.set_xlabel('Distance from center (in multiples of Schwarzschild Radii)')
    ax2.set_ylabel('Metric Value')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
