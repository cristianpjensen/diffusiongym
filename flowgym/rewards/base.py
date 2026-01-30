"""Base reward classes and interfaces for flowgym."""

from abc import ABC, abstractmethod
from typing import Any, Generic

import torch

from flowgym.types import D


class Reward(ABC, Generic[D]):
    """Abstract base class for all rewards."""

    @abstractmethod
    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reward and validity for the given input x."""


class DummyReward(Reward[D]):
    """Dummy reward that always returns zero."""

    def __call__(self, sample: D, latent: D, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(len(sample), device=sample.device),
            torch.ones(len(sample), device=sample.device),
        )
