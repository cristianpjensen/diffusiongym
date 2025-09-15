"""Base reward classes and interfaces for flow_gym."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import torch

T = TypeVar("T")


class Reward(ABC):
    """Abstract base class for all rewards."""

    @property
    @abstractmethod
    def is_differentiable(self) -> bool:
        """Whether the reward is differentiable."""

    @abstractmethod
    def __call__(self, x: Any) -> torch.Tensor:
        """Compute the reward for the given input x."""


class DifferentiableReward(Reward):
    """Reward that supports differentiation."""

    @property
    def is_differentiable(self) -> bool:
        """Whether the reward is differentiable (always True)."""
        return True

    @abstractmethod
    def gradient(self, x: T) -> T:
        """Compute the gradient of the reward with respect to x."""


class NonDifferentiableReward(Reward):
    """Reward that does not support differentiation."""

    @property
    def is_differentiable(self) -> bool:
        """Whether the reward is differentiable (always False)."""
        return False
