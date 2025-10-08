"""Optional image base models and rewards for Flow Gym."""

from .base_models.cifar import CIFARBaseModel
from .base_models.stable_diffusion import SD2BaseModel
from .rewards.compression import CompressionReward, IncompressionReward

__all__ = ["CIFARBaseModel", "CompressionReward", "IncompressionReward", "SD2BaseModel"]
