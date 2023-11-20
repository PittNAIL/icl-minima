import torch


class NMinimaFunction:
    """Vectorized implementation of the function generator that produces functions out of minima."""

    def f(
        self,
        x: torch.Tensor,
        c: torch.Tensor = torch.tensor([1.0]),
        beta: float = 2.0,
        a: float = 0.0,
        b: float = 1.0,
    ) -> torch.Tensor:
        """A function similar to the beta distribution and supported on [a + b * c, a - b * c]."""

        return (c - ((x - a) / b).abs().min(c)) ** beta

    def __call__(self, minima: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        """A method that produces functions with the given minima."""

        xs, ys = minima.T

        _scale_m = torch.cdist(xs.view(-1, 1), xs.view(-1, 1))
        _scale = _scale_m[_scale_m > 0]
        scale = 1 if _scale.nelement() == 0 else _scale.min() / 2

        m = ys.max() + 1

        out = self.f(ps.unsqueeze(1).repeat(1, len(xs)), a=xs, b=scale)

        return m + ((ys - m) * out).sum(dim=1)
