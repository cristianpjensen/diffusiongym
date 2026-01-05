"""Common noise schedules for flow matching and diffusion models."""

import torch

from flowgym.types import D

from .base import NoiseSchedule


class ConstantNoiseSchedule(NoiseSchedule[D]):
    """Constant noise schedule with fixed sigma.

    Parameters
    ----------
    sigma : float
        Constant noise level.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: D, t: torch.Tensor) -> D:
        """Constant noise schedule."""
        return self.sigma * x.ones_like()
