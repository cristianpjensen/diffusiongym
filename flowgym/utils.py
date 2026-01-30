"""Utility functions for flowgym."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator, Generic, Optional, TypeVar

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
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


T = TypeVar("T")


def index_dict(d: T, start: int, end: Optional[int] = None) -> T:
    """Recursively index into the leaves of a nested dictionary.

    Parameters
    ----------
    d : T
        Any value, if a dictionary, will be processed recursively.
    start : int
        The index to select from list/tensor leaves.
    end : Optional[int], optional
        The end index to select from list/tensor leaves, by default None.

    Returns
    -------
    T
        If d is a dictionary, returns a dictionary with the same keys and indexed leaves.
    """
    if end is None:
        idx = start
    else:
        idx = slice(start, end)

    if isinstance(d, dict):
        return {k: index_dict(v, start, end) for k, v in d.items()}  # type: ignore

    elif isinstance(d, (list, tuple, torch.Tensor)):
        return d[idx]  # type: ignore

    elif isinstance(d, (float, int, str)):
        return d

    else:
        raise TypeError(f"Unsupported leaf type: {type(d)}")


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


class FlowDataset(Dataset[tuple[D, dict[str, Any], torch.Tensor]]):
    """Dataset wrapper for flowgym data."""

    def __init__(self, data: list[D], kwargs: list[dict[str, Any]] | None, weights: list[torch.Tensor] | None):
        if len(data) == 0:
            raise ValueError("Data list is empty.")

        # Combine all data into a single object
        self.data = type(data[0]).collate(data)

        if weights is None:
            self.weights = torch.ones(len(self.data))
        else:
            self.weights = torch.cat(weights, dim=0)

        if kwargs is None:
            kwargs = [{}] * len(data)

        all_kwargs = []
        for d, k in zip(data, kwargs):
            for i in range(len(d)):
                all_kwargs.append(index_dict(k, i))

        self.kwargs: dict = default_collate(all_kwargs)  # type: ignore

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[D, dict[str, Any], torch.Tensor]:
        if self.kwargs is None:
            return self.data[idx], {}, self.weights[idx]

        return self.data[idx], index_dict(self.kwargs, idx), self.weights[idx]  # type: ignore

    def collate(self, batch):
        data_batch, kwargs_batch, weight_batch = zip(*batch)
        data_batch = type(data_batch[0]).collate(list(data_batch))
        kwargs_batch = default_collate(kwargs_batch)
        weight_batch = default_collate(weight_batch)
        return data_batch, kwargs_batch, weight_batch


def train_base_model(
    base_model: BaseModel[D],
    opt: torch.optim.Optimizer,
    data: list[D],
    kwargs: Optional[list[dict]] = None,
    weights: Optional[list[torch.Tensor]] = None,
    steps: int = 1000,
    batch_size: int = 64,
    accumulate_steps: int = 1,
    pbar: bool = False,
) -> None:
    """Trains/fine-tunes a base model.

    Parameters
    ----------
    base_model : BaseModel[D]
        The model to train.
    opt : torch.optim.Optimizer
        Optimizer to use.
    data : list[D]
        The training data.
    kwargs : list[dict]
        Keyword arguments corresponding to the data.
    weights : list[torch.Tensor]
        Training weights for the data.
    steps : int
        Number of training steps.
    batch_size : int
        Batch size.
    accumulate_steps : int
        Number of gradient accumulation steps.
    pbar : bool, default: False
        Whether to display a tqdm progress bar or not.
    """
    dataset = FlowDataset(data, kwargs, weights)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        num_workers=0,
        pin_memory=False,
    )

    base_model.train()
    opt.zero_grad()

    # Create an iterator for the dataloader
    data_iter = iter(loader)

    iterator = range(steps)
    if pbar:
        iterator = tqdm(iterator)

    loss_sum = 0.0
    grad_norm_sum = 0.0
    n_steps = 0
    for _ in iterator:
        n_steps += 1

        # Get the next batch. If the loader is exhausted, restart it.
        try:
            x1_cpu, kwargs, weight = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1_cpu, kwargs, weight = next(data_iter)

        x1_cpu: D
        x1 = x1_cpu.to(base_model.device)
        weight = weight.to(base_model.device)
        loss = (weight * base_model.train_loss(x1, **kwargs)).mean()
        loss_sum += loss.item()

        loss = loss / accumulate_steps
        loss.backward()

        if n_steps % accumulate_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(base_model.parameters(), 0.1)
            grad_norm_sum += grad_norm.item()
            opt.step()
            opt.zero_grad()

        if isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": loss_sum / n_steps, "grad_norm": grad_norm_sum / n_steps})

    base_model.eval()
