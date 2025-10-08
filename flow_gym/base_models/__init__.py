"""Base models for flow matching and diffusion."""

from .base import BaseModel
from .one_dim_gmm import OneDimensionalBaseModel

__all__ = ["BaseModel", "OneDimensionalBaseModel"]
