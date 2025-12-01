"""Binary reward for one-dimensional toy environments."""

from typing import Any

import numpy as np
import torch

from flowgym.registry import reward_registry
from flowgym.types import FGTensor

from .base import Reward


@reward_registry.register("1d/binary")
class BinaryReward(Reward[FGTensor]):
    """Binary reward for one-dimensional toy environments."""

    def __call__(self, x: FGTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the reward function at the given points."""
        result = ((x >= 0) & (x <= 1)).to(torch.float32).squeeze(-1)
        return result, torch.ones_like(result)


@reward_registry.register("1d/gaussian")
class GaussianReward(Reward[FGTensor]):
    """Gaussian reward for one-dimensional toy environments."""

    def __call__(self, x: FGTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the reward function at the given points."""
        mu = -2.5
        sigma = 0.8
        pdf = torch.exp(-0.5 * torch.square((x - mu) / sigma)) / (sigma * np.sqrt(2 * np.pi))
        result: torch.Tensor = pdf.to(torch.float32).squeeze()
        return result, torch.ones_like(result)
