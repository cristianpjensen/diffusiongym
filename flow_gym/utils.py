"""Utility functions for flow_gym."""

import os
import tempfile
from contextlib import contextmanager
from typing import Generator

import torch


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
