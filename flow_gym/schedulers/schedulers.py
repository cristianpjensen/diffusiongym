"""Common schedulers for flow matching and diffusion models."""

from typing import cast

import torch

from flow_gym.utils import FGTensor

from .base import Scheduler


class OptimalTransportScheduler(Scheduler[FGTensor]):
    r"""Optimal transport scheduler which is commonly used to train flow matching models.

    Schedule:
    .. math::
        \alpha_t = t, \quad \beta_t = 1 - t, \quad \dot{\alpha}_t = 1, \quad \dot{\beta}_t = -1.
    """

    def alpha(self, t: torch.Tensor) -> FGTensor:
        r""":math:`\alpha_t = t`."""
        return FGTensor(t)

    def alpha_dot(self, t: torch.Tensor) -> FGTensor:
        r""":math:`\dot{\alpha}_t = 1`."""
        return FGTensor(torch.ones_like(t))


class DiffusionScheduler(Scheduler[FGTensor]):
    """Scheduler for discrete-time diffusion models based on a given noise schedule.

    Parameters
    ----------
    alpha_bar : torch.Tensor
        Cumulative product of (1 - beta) values, shape (K,), where K is the number of diffusion
        steps.
    """

    def __init__(self, alpha_bar: torch.Tensor):
        super().__init__()

        self.alpha_bar = alpha_bar
        self.alpha_bar_shifted = torch.cat(
            [torch.ones(1, device=alpha_bar.device, dtype=alpha_bar.dtype), alpha_bar[:-1]], dim=0
        )
        self.K = alpha_bar.shape[0] - 1
        self.alpha_bar_dot = self.K * (self.alpha_bar_shifted - self.alpha_bar)

    def _get_index(self, t: torch.Tensor) -> torch.Tensor:
        k = ((1 - t) * self.K + 0.5).long().clamp(0, self.K)
        return cast("torch.Tensor", k)

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t that encodes the timestep."""
        return self._get_index(t)

    def alpha(self, t: torch.Tensor) -> FGTensor:
        r""":math:`\alpha_t`."""
        k = self._get_index(t)
        return FGTensor(torch.sqrt(self.alpha_bar[k]))

    def beta(self, t: torch.Tensor) -> FGTensor:
        r""":math:`\beta_t`."""
        k = self._get_index(t)
        return FGTensor(torch.sqrt(1 - self.alpha_bar[k]))

    def alpha_dot(self, t: torch.Tensor) -> FGTensor:
        r""":math:`\dot{\alpha}_t`."""
        k = self._get_index(t)
        return FGTensor(0.5 * self.alpha_bar_dot[k] / self.alpha(t))

    def beta_dot(self, t: torch.Tensor) -> FGTensor:
        r""":math:`\dot{\beta}_t`."""
        k = self._get_index(t)
        return FGTensor(-0.5 * self.alpha_bar_dot[k] / self.beta(t))
