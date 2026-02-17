"""Common schedulers for flow matching and diffusion models."""

from typing import cast

import torch

from diffusiongym.types import DDTensor
from diffusiongym.utils import append_dims

from .base import Scheduler


class OptimalTransportScheduler(Scheduler[DDTensor]):
    r"""Optimal transport scheduler which is commonly used to train flow matching models.

    Schedule:
    .. math::

        \alpha_t = t, \quad \beta_t = 1 - t, \quad \dot{\alpha}_t = 1, \quad \dot{\beta}_t = -1.
    """

    def alpha(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        r""":math:`\alpha_t = t`."""
        return DDTensor(append_dims(t, x.data.ndim))

    def alpha_dot(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        r""":math:`\dot{\alpha}_t = 1`."""
        return x.ones_like()


class CosineScheduler(Scheduler[DDTensor]):
    """Cosine scheduler."""

    def __init__(self, nu: float):
        self.nu = nu

    def alpha(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        alpha = 1 - torch.cos(0.5 * torch.pi * torch.pow(t, self.nu)).square()
        return DDTensor(append_dims(alpha, x.data.ndim))

    def alpha_dot(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        t = t.clamp(min=1e-9)
        alpha_dot = 0.5 * self.nu * torch.pi * torch.pow(t, self.nu - 1) * torch.sin(torch.pow(t, self.nu) * torch.pi)
        return DDTensor(append_dims(alpha_dot, x.data.ndim))


class DiffusionScheduler(Scheduler[DDTensor]):
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
        self.alpha_bar_shifted = torch.cat([torch.ones(1, device=alpha_bar.device, dtype=alpha_bar.dtype), alpha_bar[:-1]], dim=0)
        self.K = alpha_bar.shape[0] - 1
        self.alpha_bar_dot = self.K * (self.alpha_bar_shifted - self.alpha_bar)

    def _get_index(self, t: torch.Tensor) -> torch.Tensor:
        k = ((1 - t) * self.K + 0.5).long().clamp(0, self.K).cpu()
        return cast("torch.Tensor", k)

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t that encodes the timestep."""
        return self._get_index(t).to(t.device)

    def alpha(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        k = self._get_index(t)
        alpha = torch.sqrt(self.alpha_bar[k])
        return DDTensor(append_dims(alpha, x.data.ndim))

    def beta(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        k = self._get_index(t)
        beta = torch.sqrt(1 - self.alpha_bar[k])
        return DDTensor(append_dims(beta, x.data.ndim))

    def alpha_dot(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        k = self._get_index(t)
        alpha = torch.sqrt(self.alpha_bar[k])
        alpha_dot = 0.5 * self.alpha_bar_dot[k] / alpha
        return DDTensor(append_dims(alpha_dot, x.data.ndim))

    def beta_dot(self, x: DDTensor, t: torch.Tensor) -> DDTensor:
        k = self._get_index(t)
        beta = torch.sqrt(1 - self.alpha_bar[k])
        beta_dot = -0.5 * self.alpha_bar_dot[k] / beta
        return DDTensor(append_dims(beta_dot, x.data.ndim))
