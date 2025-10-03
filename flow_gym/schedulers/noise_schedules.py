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

    def __call__(self, x: DataType, t: torch.Tensor) -> DataType:
        """Constant noise schedule."""
        return self.sigma * x.ones_like()
