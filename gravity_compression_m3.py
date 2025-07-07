# gravity_compression_m3.py

# Use NumPy, which on Apple Silicon is highly optimized.
import numpy as np
# Import physical constants for accuracy.
from scipy.constants import G, c, k_b, hbar

# Define the main class for the theoretical model.
class CompressionModel:
    """
    Tests physical theories by modeling them as compression algorithms.
    This version is optimized for Apple Silicon (M-series chips).
    """
    def __init__(self, compression_function):
        # Stores the specific theory (function) to be tested.
        self.compression_function = compression_function

    def get_source_information(self, mass):
        # Calculates the target information value based on the source mass.
        info_scaling_factor = (4 * np.pi * G * k_b) / (hbar * c)
        return info_scaling_factor * mass**2

    def get_geometric_information_capacity(self, r_grid, g_rr_proposed):
        # Calculates the information capacity of the geometry from its horizon area.
        if np.max(g_rr_proposed) < 1e5: # Check if a horizon formed.
            return 0.0
        
        horizon_index = np.argmax(g_rr_proposed) # Find the horizon's location.
        rs_predicted = r_grid[horizon_index] # Get the predicted radius.
        area = 4 * np.pi * rs_predicted**2 # Calculate the horizon's area.
        entropy = (k_b * c**3 * area) / (4 * G * hbar) # Convert area to entropy.
        return entropy

    def calculate_information_loss(self, r_grid, mass, **kwargs):
        # The main evaluation function.
        # It generates a metric using the provided theory.
        g_tt_proposed, g_rr_proposed = self.compression_function(r_grid, mass, **kwargs)
        
        # Calculate the source information to be encoded.
        source_info = self.get_source_information(mass)
        # Calculate the information capacity of the resulting geometry.
        geom_info_capacity = self.get_geometric_information_capacity(r_grid, g_rr_proposed)
        
        # The loss is the squared difference. Zero is a perfect score.
        loss = (geom_info_capacity - source_info)**2
        return loss
