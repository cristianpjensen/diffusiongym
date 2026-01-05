"""Base classes for schedulers of flow matching models."""

from abc import ABC, abstractmethod
from typing import Generic

import torch

from flowgym.types import D


class NoiseSchedule(ABC, Generic[D]):
    """Abstract base class for noise schedules."""

    @abstractmethod
    def __call__(self, x: D, t: torch.Tensor) -> D:
        r"""Compute the noise level at time t.

        Can be overwritten if the noise schedule is data-dependent.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        sigma_t : D, same data shape as x
            Values of :math:`\sigma(t)` at the given times.
        """
        ...


class Scheduler(ABC, Generic[D]):
    r"""Abstract base class for schedulers of flow matching models.

    Generally :math:`\beta_t = 1-\alpha_t`, but this can be re-defined. Furthermore, generally we
    are interested in a memoryless noise schedule, which is the default of `noise_schedule` (i.e.,
    :math:`\sigma`), however this can also be re-defined.
    """

    @property
    def noise_schedule(self) -> NoiseSchedule[D]:
        """Get the current noise schedule."""
        if not hasattr(self, "_noise_schedule"):
            self._noise_schedule: NoiseSchedule[D] = MemorylessNoiseSchedule(self)

        return self._noise_schedule

    @noise_schedule.setter
    def noise_schedule(self, schedule: NoiseSchedule[D]) -> None:
        """Set the noise schedule. Defaults to the memoryless noise schedule."""
        self._noise_schedule = schedule

    @abstractmethod
    def alpha(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\alpha_t`.

        Can be overwritten if :math:`\alpha_t` is data-dependent.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        alpha_t : D, same data shape as x
            Values of :math:`\alpha_t` at the given times.
        """
        ...

    @abstractmethod
    def alpha_dot(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\dot{\alpha}_t`.

        Can be overwritten if :math:`\dot{\alpha}_t` is data-dependent.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        alpha_dot_t : D, same data shape as x
            Values of :math:`\dot{\alpha}_t` at the given times.
        """
        ...

    def beta(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\beta_t`.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        beta_t : D, same data shape as x
            Values of :math:`\beta_t` at the given times.
        """
        return 1 - self.alpha(x, t)

    def beta_dot(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\dot{\beta}_t`.

        Can be overwritten if :math:`\dot{\beta}_t` is data-dependent.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        beta_dot_t : D, same data shape as x
            Values of :math:`\dot{\beta}_t` at the given times.
        """
        return -self.alpha_dot(x, t)

    def sigma(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\sigma(t)` noise schedule.

        Can be overwritten if :math:`\sigma(t)` is data-dependent.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        sigma_t : D, same data shape as x
            Values of :math:`\sigma(t)` at the given times.
        """
        return self.noise_schedule(x, t)

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t.

        Defaults to t, but could be different if using a different time parameterization.
        """
        return t

    def kappa(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\kappa_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        return self.alpha_dot(x, t) / self.alpha(x, t)

    def eta(self, x: D, t: torch.Tensor) -> D:
        r""":math:`\eta_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        alpha = self.alpha(x, t)
        alpha_dot = self.alpha_dot(x, t)
        beta = self.beta(x, t)
        beta_dot = self.beta_dot(x, t)
        return beta * ((alpha_dot / alpha) * beta - beta_dot)


class MemorylessNoiseSchedule(NoiseSchedule[D]):
    r"""Memoryless noise schedule based on the scheduler's eta function.

    This schedule ensures that :math:`x_0` and :math:`x_1` are independent, which is necessary for
    unbiased generative optimization.

    Parameters
    ----------
    scheduler : Scheduler
        Scheduler to use for computing :math:`\eta_t`.
    """

    def __init__(self, scheduler: Scheduler[D]):
        self.scheduler = scheduler

    def __call__(self, x: D, t: torch.Tensor) -> D:
        r"""Compute the noise level at time t.

        This is given by :math:`\sigma(t) = \sqrt{2 \eta(t)}`.

        Parameters
        ----------
        x : D
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        sigma_t : D, same data shape as x
            Values of :math:`\sigma(t)` at the given times.
        """
        return (2 * self.scheduler.eta(x, t)) ** 0.5
