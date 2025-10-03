"""Base classes for schedulers of flow matching models."""

from abc import ABC, abstractmethod
from typing import Generic, Optional

import torch

from flow_gym.utils import DataType


class NoiseSchedule(ABC, Generic[DataType]):
    """Abstract base class for noise schedules."""

    @abstractmethod
    def __call__(self, x: DataType, t: torch.Tensor) -> DataType:
        """Compute the noise level at time t."""


class Scheduler(ABC, Generic[DataType]):
    r"""Abstract base class for schedulers of flow matching models.

    Generally :math:`\beta_t = 1-\alpha_t`, but this can be re-defined. Furthermore, generally we
    are interested in a memoryless noise schedule, which is the default of `noise_schedule` (i.e.,
    :math:`\sigma`), however this can also be re-defined.
    """

    def __init__(self, noise_schedule: Optional[NoiseSchedule[DataType]] = None):
        if noise_schedule is None:
            noise_schedule = MemorylessNoiseSchedule(self)

        self._noise_schedule = noise_schedule

    @property
    def noise_schedule(self) -> NoiseSchedule[DataType]:
        """Get the current noise schedule."""
        return self._noise_schedule

    @noise_schedule.setter
    def noise_schedule(self, schedule: NoiseSchedule[DataType]) -> None:
        """Set the noise schedule. Defaults to the memoryless noise schedule."""
        self._noise_schedule = schedule

    @abstractmethod
    def alpha(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\alpha_t`."""

    @abstractmethod
    def alpha_dot(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\dot{\alpha}_t`."""

    def beta(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\beta_t = 1-\alpha_t`."""
        return 1 - self.alpha(x, t)

    def beta_dot(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\dot{\beta}_t = -\dot{\alpha}_t`."""
        return -self.alpha_dot(x, t)

    def sigma(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\sigma(t)` noise schedule."""
        return self.noise_schedule(x, t)

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t.

        Defaults to t, but could be different if using a different time parameterization.
        """
        return t

    def kappa(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\kappa_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        return self.alpha_dot(x, t) / self.alpha(x, t)

    def eta(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\eta_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        alpha = self.alpha(x, t)
        alpha_dot = self.alpha_dot(x, t)
        beta = self.beta(x, t)
        beta_dot = self.beta_dot(x, t)
        return beta * ((alpha_dot / alpha) * beta - beta_dot)


class MemorylessNoiseSchedule(NoiseSchedule[DataType]):
    r"""Memoryless noise schedule based on the scheduler's eta function.

    This schedule ensures that :math:`x_0` and :math:`x_1` are independent, which is necessary for
    unbiased generative optimization.

    Parameters
    ----------
    scheduler : Scheduler
        Scheduler to use for computing :math:`\eta_t`.

    """

    def __init__(self, scheduler: Scheduler[DataType]):
        self.scheduler = scheduler

    def __call__(self, x: DataType, t: torch.Tensor) -> DataType:
        """Memoryless noise schedule."""
        return (2 * self.scheduler.eta(x, t)) ** 0.5
