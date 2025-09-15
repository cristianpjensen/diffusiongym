"""Base models for flow matching and diffusion."""

from .base import BaseModel
from .cifar import CIFARBaseModel

__all__ = [
    "BaseModel",
    "CIFARBaseModel",
]
