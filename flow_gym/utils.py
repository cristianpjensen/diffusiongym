"""Utility functions for flow_gym."""

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
