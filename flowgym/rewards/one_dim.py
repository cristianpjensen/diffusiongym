"""Binary reward for one-dimensional toy environments."""

from typing import Any

import numpy as np
import torch

from flowgym.registry import reward_registry
from flowgym.types import FlowTensor

from .base import Reward


@reward_registry.register("1d/binary")
class BinaryReward(Reward[FlowTensor]):
    """Binary reward for one-dimensional toy environments."""

    def __call__(self, sample: FlowTensor, latent: FlowTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the reward function at the given points."""
        result = ((sample.data >= 0) & (sample.data <= 1)).to(torch.float32).squeeze(-1)
        return result, torch.ones_like(result)


@reward_registry.register("1d/gaussian")
class GaussianReward(Reward[FlowTensor]):
    """Gaussian reward for one-dimensional toy environments."""

    def __call__(self, sample: FlowTensor, latent: FlowTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the reward function at the given points."""
        mu = -2.5
        sigma = 0.8
        pdf = torch.exp(-0.5 * torch.square((sample.data - mu) / sigma)) / (sigma * np.sqrt(2 * np.pi))
        result: torch.Tensor = pdf.to(torch.float32).squeeze()
        return result, torch.ones_like(result)
