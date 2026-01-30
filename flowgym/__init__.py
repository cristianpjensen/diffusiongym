"""flowgym package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flowgym")
except PackageNotFoundError:
    __version__ = "0.0.0"

from flowgym.base_models import BaseModel
from flowgym.environments import (
    EndpointEnvironment,
    Environment,
    EpsilonEnvironment,
    Sample,
    ScoreEnvironment,
    VelocityEnvironment,
)
from flowgym.make import construct_env, make
from flowgym.registry import base_model_registry, reward_registry
from flowgym.rewards import Reward
from flowgym.schedulers import (
    ConstantNoiseSchedule,
    CosineScheduler,
    DiffusionScheduler,
    MemorylessNoiseSchedule,
    NoiseSchedule,
    OptimalTransportScheduler,
    Scheduler,
)
from flowgym.types import D, FlowMixin, FlowTensor
from flowgym.utils import train_base_model

__all__ = [
    "BaseModel",
    "ConstantNoiseSchedule",
    "CosineScheduler",
    "D",
    "DiffusionScheduler",
    "EndpointEnvironment",
    "Environment",
    "EpsilonEnvironment",
    "FlowMixin",
    "FlowTensor",
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
