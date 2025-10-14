"""Common schedulers for flow matching and diffusion models."""

from typing import cast

import torch

from flow_gym.types import FGTensor
from flow_gym.utils import append_dims

from .base import Scheduler


class OptimalTransportScheduler(Scheduler[FGTensor]):
    r"""Optimal transport scheduler which is commonly used to train flow matching models.

    Schedule:
    .. math::

        \alpha_t = t, \quad \beta_t = 1 - t, \quad \dot{\alpha}_t = 1, \quad \dot{\beta}_t = -1.
    """

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(-1)

    def alpha(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\alpha_t = t`."""
        return FGTensor(append_dims(self._alpha(t), x.ndim))

    def _alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t).unsqueeze(-1)

    def alpha_dot(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\dot{\alpha}_t = 1`."""
        return FGTensor(append_dims(self._alpha_dot(t), x.ndim))


class CosineScheduler(Scheduler[FGTensor]):
    """Cosine scheduler."""

    def __init__(self, nu: float):
        self.nu = nu

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = 1 - torch.cos((t**self.nu) * torch.pi / 2).square()
        return result

    def alpha(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\alpha_t`."""
        return FGTensor(append_dims(self._alpha(t), x.ndim))

    def _alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = (
            self.nu
            * torch.pi
            * (t ** (self.nu - 1))
            * torch.sin((t**self.nu) * torch.pi / 2)
            * torch.cos((t**self.nu) * torch.pi / 2)
        )
        return result

    def alpha_dot(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\dot{\alpha}_t`."""
        return FGTensor(append_dims(self._alpha_dot(t), x.ndim))


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
        k = ((1 - t) * self.K + 0.5).long().clamp(0, self.K).cpu()
        return cast("torch.Tensor", k)

    def model_input(self, t: torch.Tensor) -> torch.Tensor:
        """Input to the model at time t that encodes the timestep."""
        return self._get_index(t).to(t.device)

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        k = self._get_index(t)
        return torch.sqrt(self.alpha_bar[k]).unsqueeze(-1)

    def alpha(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\alpha_t`."""
        return FGTensor(append_dims(self._alpha(t), x.ndim))

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        k = self._get_index(t)
        return torch.sqrt(1 - self.alpha_bar[k]).unsqueeze(-1)

    def beta(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\beta_t`."""
        return FGTensor(append_dims(self._beta(t), x.ndim))

    def _alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        k = self._get_index(t)
        return 0.5 * self.alpha_bar_dot[k] / self._alpha(t)

    def alpha_dot(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\dot{\alpha}_t`."""
        return FGTensor(append_dims(self._alpha_dot(t), x.ndim))

    def _beta_dot(self, t: torch.Tensor) -> torch.Tensor:
        k = self._get_index(t)
        return -0.5 * self.alpha_bar_dot[k] / self._beta(t)

    def beta_dot(self, x: FGTensor, t: torch.Tensor) -> FGTensor:
        r""":math:`\dot{\beta}_t`."""
        return FGTensor(append_dims(self._beta_dot(t), x.ndim))
