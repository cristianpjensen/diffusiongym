"""Utility functions for flow_gym."""

from typing import Protocol, Self, TypeVar, Union

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


class ArithmeticType(Protocol):
    """Protocol for types that support basic arithmetic operations."""

    def __add__(self, other: Union[Self, float, torch.Tensor]) -> Self: ...
    def __sub__(self, other: Union[Self, float, torch.Tensor]) -> Self: ...
    def __mul__(self, other: Union[Self, float, torch.Tensor]) -> Self: ...
    def __truediv__(self, other: Union[Self, float, torch.Tensor]) -> Self: ...
    def __neg__(self) -> Self: ...
    def __radd__(self, other: Union[Self, float, torch.Tensor]) -> Self: ...
    def __rsub__(self, other: Union[Self, float, torch.Tensor]) -> Self: ...
    def __rmul__(self, other: Union[Self, float, torch.Tensor]) -> Self: ...
    def __pow__(self, power: float) -> Self: ...

    def randn_like(self) -> Self:
        """Generate Gaussian noise with the same shape and type as self."""
        ...

    def ones_like(self) -> Self:
        """Generate ones with the same shape and type as self."""
        ...

    def zeros_like(self) -> Self:
        """Generate zeros with the same shape and type as self."""
        ...

    def batch_sum(self) -> torch.Tensor:
        """Sum over all dimensions except the first (batch) dimension.

        Returns
        -------
        sum : torch.Tensor, shape (batch_size,)
        """
        ...


DataType = TypeVar("DataType", bound=ArithmeticType)


class FGTensor(torch.Tensor):
    """A torch.Tensor subclass that supports required factory methods."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> "FGTensor":
        """Create a new FlowGymTensor from a torch.Tensor."""
        if isinstance(tensor, FGTensor):
            return tensor

        return tensor.as_subclass(FGTensor)

    def _wrap_result(self, result: torch.Tensor) -> "FGTensor":
        """Wrap a tensor result as FlowGymTensor."""
        if isinstance(result, FGTensor):
            return result
        return result.as_subclass(FGTensor)

    def __add__(self, other: Union["FGTensor", float, torch.Tensor]) -> "FGTensor":
        result = super().__add__(other)
        return self._wrap_result(result)

    def __sub__(self, other: Union["FGTensor", float, torch.Tensor]) -> "FGTensor":
        result = super().__sub__(other)
        return self._wrap_result(result)

    def __mul__(self, other: Union["FGTensor", float, torch.Tensor]) -> "FGTensor":
        result = super().__mul__(other)
        return self._wrap_result(result)

    def __truediv__(self, other: Union["FGTensor", float, torch.Tensor]) -> "FGTensor":
        result = super().__truediv__(other)
        return self._wrap_result(result)

    def __neg__(self) -> "FGTensor":
        result = super().__neg__()
        return self._wrap_result(result)

    def __radd__(self, other: Union["FGTensor", float, torch.Tensor]) -> "FGTensor":
        result = super().__radd__(other)
        return self._wrap_result(result)

    def __rsub__(self, other: Union["FGTensor", float, torch.Tensor]) -> "FGTensor":
        result = super().__rsub__(other)
        return self._wrap_result(result)

    def __rmul__(self, other: Union["FGTensor", float, torch.Tensor]) -> "FGTensor":
        result = super().__rmul__(other)
        return self._wrap_result(result)

    def __pow__(self, power: float) -> "FGTensor":
        result = super().__pow__(power)
        return self._wrap_result(result)

    def batch_sum(self) -> torch.Tensor:
        """Sum over all dimensions except the first (batch) dimension."""
        return self.sum(dim=tuple(range(1, self.ndim)))

    def randn_like(self) -> "FGTensor":
        """Generate random normal noise with the same shape and type as self."""
        return self._wrap_result(torch.randn_like(self))

    def ones_like(self) -> "FGTensor":
        """Generate ones with the same shape and type as self."""
        return self._wrap_result(torch.ones_like(self))

    def zeros_like(self) -> "FGTensor":
        """Generate zeros with the same shape and type as self."""
        return self._wrap_result(torch.zeros_like(self))
