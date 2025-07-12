import torch

Tensor = torch.Tensor  # Type alias for brevity
# <reason>chain: Type alias unchanged for brevity.</reason>


class GravitationalTheory:
    """
    Abstract base class for all gravitational theories.
    <reason>This class defines a common interface (`get_metric`) that all theories must implement. This polymorphic design allows the integrator to treat any theory identically, simplifying the simulation logic and making the framework easily extensible.</reason>
    """
    # New: Add category and sweep as class variables for auto-categorization and parameter sweep
    category = "classical"  # Default; subclasses can override to "quantum" or other
    sweep = None            # Default; subclasses can override with dict of param: values

    # New: Cache-related attributes
    cacheable = False       # Default to not cacheable; subclasses override to True if suitable for caching
    def get_cache_tag(self, N_STEPS, precision_tag, r0_tag):
        """
        Returns a unique tag for caching this theory's trajectory.
        Subclasses should override to include parameters, e.g., return f"{self.name}_alpha{self.alpha}"
        """
        base = self.name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(".", "_")
        return f"{base}_{N_STEPS}_{precision_tag}_r{r0_tag}"

    def __init__(self, name: str) -> None:
        self.name = name
    # <reason>chain: Init name; no change.</reason>

    def get_metric(self, r, M_param, C_param, G_param):
        """Calculates the metric components (g_tt, g_rr, g_φφ, g_tφ) for a given radius."""
        raise NotImplementedError
    # <reason>chain: Abstract method; no change.</reason> 