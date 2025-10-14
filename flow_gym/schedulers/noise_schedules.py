"""Common noise schedules for flow matching and diffusion models."""

import torch

from flow_gym.types import DataType

from .base import NoiseSchedule


class ConstantNoiseSchedule(NoiseSchedule[DataType]):
    """Constant noise schedule with fixed sigma.

    Parameters
    ----------
    sigma : float
        Constant noise level.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def noise(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise level at time t."""
        return self.sigma * torch.ones_like(t).unsqueeze(-1)

    def __call__(self, x: DataType, t: torch.Tensor) -> DataType:
        """Constant noise schedule."""
        return self.sigma * x.ones_like()
