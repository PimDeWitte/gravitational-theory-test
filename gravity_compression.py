# gravity_compression.py

import numpy as np
from scipy.constants import G, c, k_b, hbar

class CompressionModel:
    """
    Tests different 'compression algorithms' to see if they produce the
    correct spacetime geometry by minimizing information loss.
    """
    def __init__(self, compression_function):
        """
        Args:
            compression_function (callable): A function that takes (r, M, **kwargs) and
                                             proposes a spacetime metric (g_tt, g_rr).
                                             This is your "compression algorithm".
        """
        self.compression_function = compression_function

    def get_source_information(self, mass):
        """
        Postulate: The fundamental information content of a source is
        proportional to its mass-energy. E=mc^2.
        """
        # This scaling factor is chosen to match the units and form of the
        # Bekenstein-Hawking entropy formula when S is proportional to M^2.
        info_scaling_factor = (4 * np.pi * G * k_b) / (hbar * c)
        return info_scaling_factor * mass**2

    def get_geometric_information_capacity(self, r_grid, g_rr_proposed):
        """
        Calculates the information capacity (entropy) of a given geometry
        by finding its event horizon area.
        """
        # The horizon is where g_rr -> infinity. We find its location
        # by looking for the maximum value in our proposed g_rr array.
        # If no significant peak is found, we assume no horizon formed.
        if np.max(g_rr_proposed) < 1e5:
            return 0.0
        
        horizon_index = np.argmax(g_rr_proposed)
        rs_predicted = r_grid[horizon_index]
        
        area = 4 * np.pi * rs_predicted**2
        
        # Bekenstein-Hawking entropy formula: S = (k_b * A) / (4 * l_p^2)
        entropy = (k_b * c**3 * area) / (4 * G * hbar)
        return entropy

    def calculate_information_loss(self, r_grid, mass, **kwargs):
        """
        The core of the model. It calculates how well the proposed geometry
        (the compressed file) represents the source information.
        Loss = (Capacity of Geometry - Information of Source)^2
        """
        # Generate the proposed metric using your compression algorithm
        g_tt_proposed, g_rr_proposed = self.compression_function(r_grid, mass, **kwargs)
        
        source_info = self.get_source_information(mass)
        geom_info_capacity = self.get_geometric_information_capacity(r_grid, g_rr_proposed)
        
        # The loss is the squared difference. A perfect theory has a loss of 0.
        loss = (geom_info_capacity - source_info)**2
        return loss
