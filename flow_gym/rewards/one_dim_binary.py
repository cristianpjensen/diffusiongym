"""Binary reward for one-dimensional toy environments."""

import torch

from flow_gym.registry import reward_registry
from flow_gym.types import FGTensor

from .base import Reward


@reward_registry.register("1d/binary")
class BinaryReward(Reward[FGTensor]):
    """Binary reward for one-dimensional toy environments."""

    def __call__(self, x: FGTensor) -> torch.Tensor:
        """Evaluate the reward function at the given points."""
        return ((x >= 0) & (x <= 1)).to(torch.float32).squeeze(-1)
