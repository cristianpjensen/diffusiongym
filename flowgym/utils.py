"""Utility functions for flow_gym."""

import os
import tempfile
from contextlib import contextmanager
from typing import Any, Generator, Generic

import torch
from torch import nn

from flowgym.schedulers import NoiseSchedule
from flowgym.types import DataType


def append_dims(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Match the number of dimensions of x to ndim by adding dimensions at the end.

    Parameters
    ----------
    x : torch.Tensor, shape (*shape)
        The input tensor.

    ndim : int
        The target number of dimensions.

    Returns
    -------
    x : torch.Tensor, shape (*shape, 1, ..., 1)
        The reshaped tensor with ndim dimensions.
    """
    if x.ndim > ndim:
        return x

    shape = x.shape + (1,) * (ndim - x.ndim)
    return x.view(shape)


@contextmanager
def temporary_workdir() -> Generator[str, None, None]:
    """Context manager that runs code in a fresh temporary directory.

    When exiting the context, it returns to the original working directory and deletes the temporary
    folder.
    """
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            yield tmp
        finally:
            os.chdir(old_cwd)


class ValuePolicy(nn.Module, Generic[DataType]):
    r"""Policy based on a value function, :math:`u(x, t) = -\sigma(t) \nabla_x V(x, t)`.

    Parameters
    ----------
    value_network : nn.Module
        The value function network, :math:`V(x, t)`.

    noise_schedule : NoiseSchedule
        The noise schedule, :math:`\sigma(t)`.
    """

    def __init__(self, value_network: nn.Module, noise_schedule: NoiseSchedule[DataType]) -> None:
        super().__init__()
        self.value_network = value_network
        self.noise_schedule = noise_schedule

    @torch.enable_grad()  # type: ignore[no-untyped-call]
    def forward(self, x: DataType, t: torch.Tensor, **kwargs: Any) -> DataType:
        """Compute control action based on value function gradient."""
        x = x.with_requires_grad()
        value_pred = self.value_network(x, t, **kwargs)
        sigma = self.noise_schedule(x, t)
        control: DataType = -sigma * x.gradient(value_pred)
        return control
