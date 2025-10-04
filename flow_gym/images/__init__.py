"""Optional image base models and rewards for Flow Gym."""

from .base_models.cifar import CIFARBaseModel
from .rewards.compression import CompressionReward, IncompressionReward

__all__ = ["CIFARBaseModel", "CompressionReward", "IncompressionReward"]
