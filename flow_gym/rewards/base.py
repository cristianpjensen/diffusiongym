"""Base reward classes and interfaces for flow_gym."""

from abc import ABC, abstractmethod
from typing import Generic

import torch

from flow_gym.types import DataType


class Reward(ABC, Generic[DataType]):
    """Abstract base class for all rewards."""

    @abstractmethod
    def __call__(self, x: DataType) -> torch.Tensor:
        """Compute the reward for the given input x."""
