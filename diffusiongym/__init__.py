"""Diffusion Gym package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("diffusiongym")
except PackageNotFoundError:
    __version__ = "0.0.0"

from diffusiongym.base_models import BaseModel
from diffusiongym.environments import (
    EndpointEnvironment,
    Environment,
    EpsilonEnvironment,
    Sample,
    ScoreEnvironment,
    VelocityEnvironment,
)
from diffusiongym.make import construct_env, make
from diffusiongym.registry import base_model_registry, reward_registry
from diffusiongym.rewards import DummyReward, Reward
from diffusiongym.schedulers import (
    ConstantNoiseSchedule,
    CosineScheduler,
    DiffusionScheduler,
    MemorylessNoiseSchedule,
    NoiseSchedule,
    OptimalTransportScheduler,
    Scheduler,
)
from diffusiongym.types import D, DDMixin, DDTensor
from diffusiongym.utils import train_base_model

__all__ = [
    "BaseModel",
    "ConstantNoiseSchedule",
    "CosineScheduler",
    "D",
    "DDMixin",
    "DDTensor",
    "DiffusionScheduler",
    "DummyReward",
    "EndpointEnvironment",
    "Environment",
    "EpsilonEnvironment",
    "MemorylessNoiseSchedule",
    "NoiseSchedule",
    "OptimalTransportScheduler",
    "Reward",
    "Sample",
    "Scheduler",
    "ScoreEnvironment",
    "VelocityEnvironment",
    "base_model_registry",
    "construct_env",
    "make",
    "reward_registry",
    "train_base_model",
]
