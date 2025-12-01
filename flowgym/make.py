"""Factory function for creating flow_gym environments."""

from typing import TYPE_CHECKING, Any, Literal, Optional, overload

import torch

from flowgym.environments import (
    EndpointEnvironment,
    Environment,
    EpsilonEnvironment,
    ScoreEnvironment,
    VelocityEnvironment,
)
from flowgym.registry import base_model_registry, reward_registry
from flowgym.types import FGTensor

if TYPE_CHECKING:
    from flowgym.molecules.types import FGGraph


# Overloads for image-based models
@overload
def make(
    base_model: Literal["images/cifar", "images/sd2"],
    reward: str,
    discretization_steps: int,
    reward_scale: float = 1.0,
    device: Optional[torch.device | str] = None,
    base_model_kwargs: Optional[dict[str, Any]] = None,
    reward_kwargs: Optional[dict[str, Any]] = None,
) -> Environment[FGTensor]: ...


# Overload for molecule-based models
@overload
def make(
    base_model: Literal["molecules/flowmol_qm9", "molecules/flowmol_geom"],
    reward: str,
    discretization_steps: int,
    reward_scale: float = 1.0,
    device: Optional[torch.device | str] = None,
    base_model_kwargs: Optional[dict[str, Any]] = None,
    reward_kwargs: Optional[dict[str, Any]] = None,
) -> Environment["FGGraph"]: ...


# General overload for any string
@overload
def make(
    base_model: str,
    reward: str,
    discretization_steps: int,
    reward_scale: float = 1.0,
    device: Optional[torch.device | str] = None,
    base_model_kwargs: Optional[dict[str, Any]] = None,
    reward_kwargs: Optional[dict[str, Any]] = None,
) -> Environment[Any]: ...


def make(
    base_model: str,
    reward: str,
    discretization_steps: int,
    reward_scale: float = 1.0,
    device: Optional[torch.device | str] = None,
    base_model_kwargs: Optional[dict[str, Any]] = None,
    reward_kwargs: Optional[dict[str, Any]] = None,
) -> Environment[Any]:
    """Create a flow_gym environment from registered base models and rewards.

    Parameters
    ----------
    base_model : str
        The ID of the base model to use (e.g., "images/cifar", "molecules/flowmol").

    reward : str
        The ID of the reward function to use (e.g., "images/compression",
        "molecules/dipole_moment").

    discretization_steps : int
        The number of discretization steps to use when sampling trajectories.

    reward_scale : float, default: 1.0
        Scaling factor for the terminal reward function.

    device : torch.device, default: cpu
        The device to run the base model on.

    base_model_kwargs : dict[str, Any], default: {}
        Keyword arguments to pass to the base model constructor.

    reward_kwargs : dict[str, Any], default: {}
        Keyword arguments to pass to the reward constructor.

    Returns
    -------
    env : Environment
        The created environment.

    Raises
    ------
    KeyError
        If the base_model or reward ID is not registered.

    ValueError
        If the base_model and reward are incompatible (e.g., mixing images and molecules),
        or if env_type is not supported.

    Examples
    --------
    >>> import flow_gym
    >>> env = flow_gym.make(
    ...     base_model="images/sd2",
    ...     reward="images/compression",
    ...     discretization_steps=100,
    ...     base_model_kwargs={"cfg_scale": 6.5},
    ...     reward_kwargs={"quality_level": 65},
    ... )
    """
    base_model_kwargs = base_model_kwargs or {}
    reward_kwargs = reward_kwargs or {}

    # Validate compatibility
    base_domain = base_model.split("/")[0] if "/" in base_model else None
    reward_domain = reward.split("/")[0] if "/" in reward else None

    if base_domain and reward_domain and base_domain != reward_domain:
        raise ValueError(
            f"Incompatible base_model and reward domains: '{base_model}' ({base_domain}) "
            f"and '{reward}' ({reward_domain}). They must be from the same domain "
            f"(e.g., both 'images' or both 'molecules')."
        )

    # Get registry entries
    base_model_entry = base_model_registry.get(base_model)
    reward_entry = reward_registry.get(reward)

    # Instantiate base model and reward
    base_model_instance = base_model_entry.instantiate(device=device, **base_model_kwargs)
    reward_instance = reward_entry.instantiate(**reward_kwargs)

    # Create environment based on type
    env_classes: dict[str, type[Environment[Any]]] = {
        "epsilon": EpsilonEnvironment,
        "endpoint": EndpointEnvironment,
        "score": ScoreEnvironment,
        "velocity": VelocityEnvironment,
    }

    # Determine environment class from base model's output type
    env_type = base_model_instance.output_type
    if env_type not in env_classes:
        raise ValueError(f"Any env_type: {env_type}. Available: {', '.join(env_classes.keys())}")

    env_class = env_classes[env_type]
    return env_class(base_model_instance, reward_instance, discretization_steps, reward_scale)
