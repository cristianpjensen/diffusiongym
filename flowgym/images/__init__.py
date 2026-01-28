"""Image base models and rewards for Flow Gym."""

from .base_models.cifar import CIFARBaseModel
from .base_models.dit import DiTBaseModel
from .base_models.stable_diffusion import SD2BaseModel, SD15BaseModel, StableDiffusionBaseModel
from .rewards import AestheticReward, CompressionReward, IncompressionReward

__all__ = [
    "AestheticReward",
    "CIFARBaseModel",
    "CompressionReward",
    "DiTBaseModel",
    "IncompressionReward",
    "SD2BaseModel",
    "SD15BaseModel",
    "StableDiffusionBaseModel",
]
