"""Base reward classes and interfaces for flowgym."""

from abc import ABC, abstractmethod
from typing import Any, Generic

import torch

from flowgym.types import D


class Reward(ABC, Generic[D]):
    """Abstract base class for all rewards."""

    # Indicates whether the reward operates in latent space or on sample space (e.g., pixel space
    # for latent diffusion models)
    latent_space: bool = False

    @abstractmethod
    def __call__(self, x: D, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reward and validity for the given input x."""
