"""Common noise schedules for flow matching and diffusion models."""

import torch

from flow_gym.utils import DataType

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

    def __call__(self, t: torch.Tensor) -> DataType:
        """Constant noise schedule."""
        raise NotImplementedError
        # return self.sigma * torch.ones_like(t)


class LinearNoiseSchedule(NoiseSchedule[DataType]):
    """Linear noise schedule between sigma_start and sigma_end.

    Parameters
    ----------
    sigma_start : float
        Starting noise level at t=0.
    sigma_end : float
        Ending noise level at t=1.
    """

    def __init__(self, sigma_start: float, sigma_end: float):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end

    def __call__(self, t: torch.Tensor) -> DataType:
        """Linear interpolation between sigma_start and sigma_end."""
        raise NotImplementedError
        # return self.sigma_start + t * (self.sigma_end - self.sigma_start)
