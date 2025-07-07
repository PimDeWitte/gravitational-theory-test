# gravity_compression_m3.py

import numpy as np
from scipy.constants import G, c

class CompressionModel:
    """
    Tests physical theories by comparing the predicted event horizon radius
    to the true Schwarzschild Radius, ensuring numerical stability.
    """
    def __init__(self, compression_function):
        self.compression_function = compression_function

    def get_true_radius(self, mass):
        """Calculates the true Schwarzschild Radius from the source mass."""
        return (2 * G * mass) / (c**2)

    def get_predicted_radius(self, r_grid, g_tt_proposed):
        """
        Finds the predicted event horizon radius by locating where g_tt = 0.
        """
        # Find the index where the absolute value of g_tt is smallest.
        horizon_index = np.argmin(np.abs(g_tt_proposed))
        
        # Check if the found point is actually a horizon (g_tt is near zero).
        if np.abs(g_tt_proposed[horizon_index]) > 1e-2:
            return 0.0 # Return 0 if no clear horizon is formed.

        # Return the radius value at the found horizon index.
        return r_grid[horizon_index]

    def calculate_loss(self, r_grid, mass, **kwargs):
        """
        Calculates the loss based on the squared difference of the predicted
        and true event horizon radii. This is numerically stable.
        """
        # Generate the proposed metric using the provided theory.
        g_tt_proposed, g_rr_proposed = self.compression_function(r_grid, mass, **kwargs)
        
        # Get the true radius based on the source mass.
        true_rs = self.get_true_radius(mass)
        # Get the predicted radius from the generated geometry.
        predicted_rs = self.get_predicted_radius(r_grid, g_tt_proposed)
        
        # The loss is the squared difference of the radii.
        loss = (predicted_rs - true_rs)**2
        return loss