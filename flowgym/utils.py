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
    steps: int,
    batch_size: int,
    opt: torch.optim.Optimizer,
    pbar: bool = False,
) -> None:
    """Trains/fine-tunes a base model.

    Parameters
    ----------
    base_model : BaseModel[D]
        The model to train.

    data : list[D]
        The training data.

    steps : int
        Number of training steps.

    batch_size : int
        Batch size.

    opt : torch.optim.Optimizer
        Optimizer to use.

    pbar : bool, default: False
        Whether to display a tqdm progress bar or not.
    """
    dataset = FlowDataset(data)
    loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=type(data[0]).collate)

    # Create an iterator for the dataloader
    data_iter = iter(loader)

    iterator = range(steps)
    if pbar:
        iterator = tqdm(iterator)

    base_model.train()
    for _ in iterator:
        # Get the next batch. If the loader is exhausted, restart it.
        try:
            x1_cpu = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1_cpu = next(data_iter)

        x1_cpu: D
        x1 = x1_cpu.to(base_model.device)
        loss = base_model.train_loss(x1).mean()

        opt.zero_grad()
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(base_model.parameters(), 0.1)
        opt.step()

        if isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": loss.item(), "grad_norm": grad_norm.item()})

    base_model.eval()
