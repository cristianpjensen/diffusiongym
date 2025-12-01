"""flowgym package."""

from flowgym.base_models import BaseModel
from flowgym.environments import (
    EndpointEnvironment,
    Environment,
    EpsilonEnvironment,
    ScoreEnvironment,
    VelocityEnvironment,
)
from flowgym.make import make
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
from flowgym.types import DataProtocol, DataType, FGTensor

__all__ = [
    "BaseModel",
    "ConstantNoiseSchedule",
    "CosineScheduler",
    "DataProtocol",
    "DataType",
    "DiffusionScheduler",
    "EndpointEnvironment",
    "Environment",
    "EpsilonEnvironment",
    "FGTensor",
    "MemorylessNoiseSchedule",
    "NoiseSchedule",
    "OptimalTransportScheduler",
    "Reward",
    "Scheduler",
    "ScoreEnvironment",
    "VelocityEnvironment",
    "base_model_registry",
    "make",
    "reward_registry",
]

try:
    from . import molecules

    HAS_MOLECULES = True
except ImportError:
    HAS_MOLECULES = False

try:
    from . import images

    HAS_IMAGES = True
except ImportError:
    HAS_IMAGES = False

if HAS_MOLECULES:
    __all__ += ["molecules"]

if HAS_IMAGES:
    __all__ += ["images"]
