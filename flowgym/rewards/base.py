"""Base reward classes and interfaces for flowgym."""

from abc import ABC, abstractmethod
from typing import Any, Generic

import torch

from flowgym.types import DataType


class Reward(ABC, Generic[DataType]):
    """Abstract base class for all rewards."""

    @abstractmethod
    def __call__(self, x: DataType, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reward and validity for the given input x."""
