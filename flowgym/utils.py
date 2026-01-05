"""Utility functions for flowgym."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Generic

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flowgym.types import D

if TYPE_CHECKING:
    from flowgym.base_models import BaseModel
    from flowgym.schedulers import NoiseSchedule


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


class ValuePolicy(nn.Module, Generic[D]):
    r"""Policy based on a value function, :math:`u(x, t) = -\sigma(t) \nabla_x V(x, t)`.

    Parameters
    ----------
    value_network : nn.Module
        The value function network, :math:`V(x, t)`.

    noise_schedule : NoiseSchedule
        The noise schedule, :math:`\sigma(t)`.
    """

    def __init__(self, value_network: nn.Module, noise_schedule: NoiseSchedule[D]) -> None:
        super().__init__()
        self.value_network = value_network
        self.noise_schedule = noise_schedule

    @torch.enable_grad()  # type: ignore[no-untyped-call]
    def forward(self, x: D, t: torch.Tensor, **kwargs: Any) -> D:
        """Compute control action based on value function gradient."""
        x = x.requires_grad()
        value_pred = self.value_network(x, t, **kwargs)
        sigma = self.noise_schedule(x, t)
        control: D = -sigma * x.gradient(value_pred)
        return control


class FlowDataset(Dataset[D]):
    """Dataset wrapper for flowgym data."""

    def __init__(self, data: list[D]):
        if len(data) == 0:
            raise ValueError("Data list is empty.")

        # Combine all data into a single object
        self.data = type(data[0]).collate(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> D:
        return self.data[idx]


def train_base_model(
    base_model: BaseModel[D],
    data: list[D],
    epochs: int,
    batch_size: int,
    opt: torch.optim.Optimizer,
    pbar: bool = False,
) -> None:
    """Trains/fine-tunes a base model.

    Parameters
    ----------
    model : BaseModel[D]
        The model to train.

    data : list[D]
        The training data.

    epochs : int
        The number of training epochs.
    """
    base_model.train()
    scheduler = base_model.scheduler

    dataset = FlowDataset(data)
    loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=type(data[0]).collate)

    iterator = range(epochs)
    if pbar:
        iterator = tqdm(iterator)

    for _ in iterator:
        total_loss = 0.0
        for batch in loader:
            x1 = batch[0]
            x1: D

            x1 = x1.to(base_model.device)
            x0 = x1.randn_like()
            t = torch.rand(len(x1), device=x1.device)

            alpha = scheduler.alpha(x1, t)
            beta = scheduler.beta(x1, t)

            xt = alpha * x1 + beta * x0
            pred = base_model(xt, t)

            if base_model.output_type == "velocity":
                alpha_dot = scheduler.alpha_dot(x1, t)
                beta_dot = scheduler.beta_dot(x1, t)
                target = alpha_dot * x1 + beta_dot * x0
            elif base_model.output_type == "endpoint":
                target = x1
            elif base_model.output_type == "epsilon":
                target = x0
            elif base_model.output_type == "score":
                target = -x0 / beta
            else:
                raise ValueError(f"Unknown output type: {base_model.output_type}")

            loss = ((pred - target) ** 2).aggregate().mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(x1)

        if isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": total_loss / len(dataset)})
