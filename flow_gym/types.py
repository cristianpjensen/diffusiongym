"""Custom types for flow_gym."""

from typing import Protocol, TypeVar, Union

import torch
from typing_extensions import Self


class DataProtocol(Protocol):
    """Protocol defining the required interface for data types in flow_gym.

    Types implementing this protocol must support arithmetic operations, factory methods
    (randn_like, ones_like, zeros_like), and batch reduction operations (batch_sum).
    """

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

    def to_device(self, device: torch.device | str) -> Self:
        """Make a copy of the data and move it to the specified device.

        Returns
        -------
        A copy of self on the specified device.
        """
        ...

    def with_requires_grad(self) -> Self:
        """Set requires_grad=True for all elements in self.

        Returns
        -------
        A copy of self with requires_grad=True.
        """
        ...

    def gradient(self, x: torch.Tensor) -> Self:
        """Compute the gradient of x with respect to self.

        Parameters
        ----------
        x : torch.Tensor
            A tensor to compute the gradient of.

        Returns
        -------
        grad : Self
            The gradient of x with respect to self.
        """
        ...


DataType = TypeVar("DataType", bound=DataProtocol)


class FGTensor(torch.Tensor):
    """A torch.Tensor subclass that supports required factory methods."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> "FGTensor":
        """Create a new FGTensor from a torch.Tensor."""
        if isinstance(tensor, FGTensor):
            return tensor

        return tensor.as_subclass(FGTensor)

    def _wrap_result(self, result: torch.Tensor) -> "FGTensor":
        """Wrap a tensor result as FGTensor."""
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

    def to_device(self, device: torch.device | str) -> "FGTensor":
        """Make a copy of the data and move it to the specified device."""
        return self._wrap_result(self.clone().to(device))

    def with_requires_grad(self) -> "FGTensor":
        """Set requires_grad=True for all elements in self."""
        return self._wrap_result(self.clone().requires_grad_(True))

    def gradient(self, x: torch.Tensor) -> "FGTensor":
        """Compute the gradient of x w.r.t. self."""
        grad = torch.autograd.grad(
            outputs=x,
            inputs=self,
            grad_outputs=torch.ones_like(x),
            create_graph=False,
            retain_graph=False,
        )[0]
        return self._wrap_result(grad)
