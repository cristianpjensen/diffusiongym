"""Common noise schedules for flow matching and diffusion models."""

import torch

from .base import NoiseSchedule, Scheduler


class MemorylessNoiseSchedule(NoiseSchedule):
    r"""Memoryless noise schedule based on the scheduler's eta function.

    This schedule ensures that :math:`x_0` and :math:`x_1` are independent, which is necessary for
    unbiased generative optimization.

    Parameters
    ----------
    scheduler : Scheduler
        Scheduler to use for computing :math:`\eta_t`.

    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Memoryless noise schedule."""
        return torch.sqrt(2 * self.scheduler.eta(t))


class ConstantNoiseSchedule(NoiseSchedule):
    """Constant noise schedule with fixed sigma.

    Parameters
    ----------
    sigma : float
        Constant noise level.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Constant noise schedule."""
        return self.sigma * torch.ones_like(t)


class LinearNoiseSchedule(NoiseSchedule):
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

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between sigma_start and sigma_end."""
        return self.sigma_start + t * (self.sigma_end - self.sigma_start)
