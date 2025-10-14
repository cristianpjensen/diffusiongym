"""flow_gym package."""

from flow_gym.base_models import BaseModel
from flow_gym.environments import (
    BaseEnvironment,
    EndpointEnvironment,
    EpsilonEnvironment,
    ScoreEnvironment,
    VelocityEnvironment,
)
from flow_gym.make import make
from flow_gym.registry import base_model_registry, reward_registry
from flow_gym.rewards import Reward
from flow_gym.schedulers import (
    ConstantNoiseSchedule,
    CosineScheduler,
    DiffusionScheduler,
    MemorylessNoiseSchedule,
    NoiseSchedule,
    OptimalTransportScheduler,
    Scheduler,
)
from flow_gym.types import DataProtocol, DataType, FGTensor

__all__ = [
    "BaseEnvironment",
    "BaseModel",
    "ConstantNoiseSchedule",
    "CosineScheduler",
    "DataProtocol",
    "DataType",
    "DiffusionScheduler",
    "EndpointEnvironment",
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
