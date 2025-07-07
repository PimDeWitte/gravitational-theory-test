# gravity_compression_gpu.py

# --- GPU MODIFICATION: Import CuPy library for GPU array computations. ---
import cupy as cp
# Import physical constants from SciPy for accurate, standardized values.
from scipy.constants import G, c, k_b, hbar

# Define the main class that encapsulates the logic of our theoretical model.
class CompressionModel:
    """
    Tests different 'compression algorithms' (physical theories) to see if they
    produce the correct spacetime geometry by minimizing information loss.
    This version is designed to run on a CUDA-enabled GPU using CuPy.
    """
    # This is the constructor for the class.
    def __init__(self, compression_function):
        """
        Args:
            compression_function (callable): A function representing a physical theory.
        """
        # Store the user-provided theory (the function) as an attribute of this object.
        self.compression_function = compression_function

    # This method calculates the theoretical information content of the source mass.
    def get_source_information(self, mass):
        """
        Postulate: The fundamental information content of a source is
        proportional to its mass-energy squared.
        """
        # This pre-factor is derived from Bekenstein-Hawking to ensure units match.
        info_scaling_factor = (4 * cp.pi * G * k_b) / (hbar * c)
        # Return the calculated source information; this is our "target" value.
        return info_scaling_factor * mass**2

    # This method calculates the information storage capacity of a given spacetime geometry.
    def get_geometric_information_capacity(self, r_grid, g_rr_proposed):
        """
        Calculates the information capacity (entropy) of a geometry
        by finding the area of its event horizon.
        """
        # Check if an event horizon formed. If not, capacity is zero.
        if cp.max(g_rr_proposed) < 1e5:
            return 0.0
        
        # Find the location of the event horizon on the GPU array.
        horizon_index = cp.argmax(g_rr_proposed)
        # Get the predicted Schwarzschild radius.
        rs_predicted = r_grid[horizon_index]
        
        # Calculate the surface area of the event horizon.
        area = 4 * cp.pi * rs_predicted**2
        
        # Use the Bekenstein-Hawking formula to convert area into entropy.
        entropy = (k_b * c**3 * area) / (4 * G * hbar)
        # Return the calculated information capacity.
        return entropy

    # This is the main method that evaluates the performance of the provided theory.
    def calculate_information_loss(self, r_grid, mass, **kwargs):
        """
        Quantifies how well the proposed geometry represents the source information.
        """
        # Execute the provided theory to generate a proposed spacetime metric on the GPU.
        g_tt_proposed, g_rr_proposed = self.compression_function(r_grid, mass, **kwargs)
        
        # Calculate the source information.
        source_info = self.get_source_information(mass)
        # Calculate the information storage capacity of the generated geometry.
        geom_info_capacity = self.get_geometric_information_capacity(r_grid, g_rr_proposed)
        
        # Calculate the squared difference (loss). A perfect theory yields zero.
        loss = (geom_info_capacity - source_info)**2
        # Return the calculated loss (still a CuPy array).
        return loss
