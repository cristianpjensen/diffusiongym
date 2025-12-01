"""Base classes for schedulers of flow matching models."""

from abc import ABC, abstractmethod
from typing import Generic

import torch

from flowgym.types import DataType


class NoiseSchedule(ABC, Generic[DataType]):
    """Abstract base class for noise schedules."""

    @abstractmethod
    def noise(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute the noise level at time t.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (n, d)
            Values of :math:`\sigma(t)` at the given times.
        """

    def __call__(self, x: DataType, t: torch.Tensor) -> DataType:
        r"""Compute the noise level at time t.

        Can be overwritten if the noise schedule is data-dependent.

        Parameters
        ----------
        x : DataType
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        sigma_t : DataType, same data shape as x
            Values of :math:`\sigma(t)` at the given times.
        """
        return x.ones_like() * self.noise(t)


class Scheduler(ABC, Generic[DataType]):
    r"""Abstract base class for schedulers of flow matching models.

    Generally :math:`\beta_t = 1-\alpha_t`, but this can be re-defined. Furthermore, generally we
    are interested in a memoryless noise schedule, which is the default of `noise_schedule` (i.e.,
    :math:`\sigma`), however this can also be re-defined.
    """

    @property
    def noise_schedule(self) -> NoiseSchedule[DataType]:
        """Get the current noise schedule."""
        if not hasattr(self, "_noise_schedule"):
            self._noise_schedule: NoiseSchedule[DataType] = MemorylessNoiseSchedule(self)

        return self._noise_schedule

    @noise_schedule.setter
    def noise_schedule(self, schedule: NoiseSchedule[DataType]) -> None:
        """Set the noise schedule. Defaults to the memoryless noise schedule."""
        self._noise_schedule = schedule

    @abstractmethod
    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\alpha_t`.

        This should be a monotonically decreasing function with :math:`\alpha_0 = 1` and
        :math:`\alpha_1 = 0`.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (n, d)
            Values of :math:`\alpha_t` at the given times.
        """

    def alpha(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\alpha_t`.

        Can be overwritten if :math:`\alpha_t` is data-dependent.

        Parameters
        ----------
        x : DataType
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        alpha_t : DataType, same data shape as x
            Values of :math:`\alpha_t` at the given times.
        """
        return x.ones_like() * self._alpha(t)

    @abstractmethod
    def _alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\dot{\alpha}_t`.

        This should be the time-derivative of :math:`\alpha_t`.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (n, d)
            Values of :math:`\dot{\alpha}_t` at the given times.
        """

    def alpha_dot(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\dot{\alpha}_t`.

        Can be overwritten if :math:`\dot{\alpha}_t` is data-dependent.

        Parameters
        ----------
        x : DataType
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        alpha_dot_t : DataType, same data shape as x
            Values of :math:`\dot{\alpha}_t` at the given times.
        """
        return x.ones_like() * self._alpha_dot(t)

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\beta_t`.

        This should be a monotonically increasing function with :math:`\beta_0 = 0` and
        :math:`\beta_1 = 1`.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (n, d)
            Values of :math:`\beta_t` at the given times.
        """
        result: torch.Tensor = 1 - self._alpha(t)
        return result

    def beta(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\beta_t`.

        Parameters
        ----------
        x : DataType
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        beta_t : DataType, same data shape as x
            Values of :math:`\beta_t` at the given times.
        """
        return 1 - self.alpha(x, t)

    def _beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\dot{\beta}_t`.

        This should be the time-derivative of :math:`\beta_t`.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        beta_dot_t : torch.Tensor, shape (n, d)
            Values of :math:`\dot{\beta}_t` at the given times.
        """
        return -self._alpha_dot(t)

    def beta_dot(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\dot{\beta}_t`.

        Can be overwritten if :math:`\dot{\beta}_t` is data-dependent.

        Parameters
        ----------
        x : DataType
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        beta_dot_t : DataType, same data shape as x
            Values of :math:`\dot{\beta}_t` at the given times.
        """
        return -self.alpha_dot(x, t)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\sigma(t)` noise schedule.

        Defaults to the memoryless noise schedule based on :math:`\eta_t`.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (n, d)
            Values of :math:`\sigma(t)` at the given times.
        """
        return self.noise_schedule.noise(t)

    def sigma(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\sigma(t)` noise schedule.

        Can be overwritten if :math:`\sigma(t)` is data-dependent.

        Parameters
        ----------
        x : DataType
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        sigma_t : DataType, same data shape as x
            Values of :math:`\sigma(t)` at the given times.
        """
        return self.noise_schedule(x, t)

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t.

        Defaults to t, but could be different if using a different time parameterization.
        """
        return t

    def _kappa(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\kappa_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u).

        This is given by :math:`\kappa_t = \dot{\alpha}_t / \alpha_t`.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (n, d)
            Values of :math:`\kappa_t` at the given times.
        """
        return self._alpha_dot(t) / self._alpha(t)

    def kappa(self, x: DataType, t: torch.Tensor) -> DataType:
        r""":math:`\kappa_t` as defined in [Adjoint Matching](https://openreview.net/forum?id=xQBRrtQM8u)."""
        return self.alpha_dot(x, t) / self.alpha(x, t)

    def _eta(self, t: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha(t)
        alpha_dot = self._alpha_dot(t)
        beta = self._beta(t)
        beta_dot = self._beta_dot(t)
        return beta * ((alpha_dot / alpha) * beta - beta_dot)

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

    def noise(self, t: torch.Tensor) -> torch.Tensor:
        r"""Compute the noise level at time t.

        This is given by :math:`\sigma(t) = \sqrt{2 \eta(t)}`.

        Parameters
        ----------
        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (n, d)
            Values of :math:`\sigma(t)` at the given times.
        """
        result: torch.Tensor = (2 * self.scheduler._eta(t)) ** 0.5
        return result

    def __call__(self, x: DataType, t: torch.Tensor) -> DataType:
        r"""Compute the noise level at time t.

        This is given by :math:`\sigma(t) = \sqrt{2 \eta(t)}`.

        Parameters
        ----------
        x : DataType
            Data tensor.

        t : torch.Tensor, shape (n,)
            Time tensor with values in [0, 1].

        Returns
        -------
        sigma_t : DataType, same data shape as x
            Values of :math:`\sigma(t)` at the given times.
        """
        return (2 * self.scheduler.eta(x, t)) ** 0.5
