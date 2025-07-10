
class FallbackTheory(GravitationalTheory):
    def __init__(self):
        super().__init__(f"FallbackTheory_{random.randint(1, 100)}")

    def get_metric(self, r: Tensor, M_param: Tensor, C_param: float, G_param: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rs = 2 * G_param * M_param / C_param**2
        m = 1 - rs / r
        return -m, 1 / (m + EPSILON), r**2, torch.zeros_like(r)
