# gravity_compression_m3.py

import numpy as np
from scipy.constants import G, c, k, hbar

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
        horizon_index = np.argmin(np.abs(g_tt_proposed))
        
        if np.abs(g_tt_proposed[horizon_index]) > 1e-2:
            return 0.0

        return r_grid[horizon_index]

    def calculate_loss(self, r_grid, mass, **kwargs):
        """
        Calculates the loss based on the squared difference of the predicted
        and true event horizon radii. This is numerically stable.
        """
        g_tt_proposed, g_rr_proposed = self.compression_function(r_grid, mass, **kwargs)
        
        true_rs = self.get_true_radius(mass)
        predicted_rs = self.get_predicted_radius(r_grid, g_tt_proposed)
        
        loss = (predicted_rs - true_rs)**2
        return loss