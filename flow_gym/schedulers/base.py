"""Base classes for schedulers of flow matching models."""

from abc import ABC, abstractmethod
from typing import Optional, cast

import torch


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the noise level at time t."""


class Scheduler(ABC):
    """Abstract base class for schedulers of flow matching models.

    Generally beta is defined to be `1-alpha`, but this can be re-defined. Furthermore, generally we
    are interested in a memoryless noise schedule, which is the default of `sigma`, however this can
    also be re-defined.
    """

    def __init__(self, noise_schedule: Optional[NoiseSchedule] = None):
        self._noise_schedule = noise_schedule

    @property
    def noise_schedule(self) -> Optional[NoiseSchedule]:
        """Get the current noise schedule."""
        return self._noise_schedule

    @noise_schedule.setter
    def noise_schedule(self, schedule: Optional[NoiseSchedule]) -> None:
        """Set the noise schedule. Use None for default memoryless behavior."""
        self._noise_schedule = schedule

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\alpha_t`."""

    @abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\dot{\alpha}_t`."""

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\beta_t = 1-\alpha_t`."""
        return cast("torch.Tensor", 1 - self.alpha(t))

    def beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\dot{\beta}_t = -\dot{\alpha}_t`."""
        return -self.alpha_dot(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Defaults to the memoryless noise schedule."""
        if self._noise_schedule is not None:
            return self._noise_schedule(t)

        # Default to memoryless noise schedule
        return torch.sqrt(2 * self.eta(t))

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t.

        Defaults to t, but could be different if using a different time parameterization.
        """
        return t

    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\kappa_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        return self.alpha_dot(t) / self.alpha(t)

    def eta(self, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\eta_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        alpha = self.alpha(t)
        alpha_dot = self.alpha_dot(t)
        beta = self.beta(t)
        beta_dot = self.beta_dot(t)
        return beta * ((alpha_dot / alpha) * beta - beta_dot)
