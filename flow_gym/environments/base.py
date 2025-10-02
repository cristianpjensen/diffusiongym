"""Base environment classes and interfaces for flow_gym."""

from abc import ABC, abstractmethod
from itertools import pairwise
from typing import Any, Generic, Optional, Protocol

import torch

from flow_gym.base_models import BaseModel
from flow_gym.rewards import Reward
from flow_gym.schedulers import Scheduler
from flow_gym.utils import DataType


class Policy(Protocol[DataType]):
    """General protocol for a policy function."""

    def __call__(self, x: DataType, t: torch.Tensor, **kwargs: dict[str, Any]) -> DataType: ...  # noqa: D102


class BaseEnvironment(ABC, Generic[DataType]):
    """Abstract base class for all environments.

    Parameters
    ----------
    base_model : BaseModel[DataType]
        The base generative model used in the environment.

    reward : Reward[DataType]
        The reward function used to compute the final reward.

    discretization_steps : int
        The number of discretization steps to use when sampling trajectories.
    """

    def __init__(
        self,
        base_model: BaseModel[DataType],
        reward: Reward[DataType],
        discretization_steps: int,
    ):
        self.base_model = base_model
        self.reward = reward
        self.discretization_steps = discretization_steps
        self._policy: Optional[Policy[DataType]] = None
        self._additive_policy: Optional[Policy[DataType]] = None

    @property
    def scheduler(self) -> Scheduler[DataType]:
        """Get the scheduler of the base model."""
        return self.base_model.scheduler

    @property
    def policy(self) -> Policy[DataType]:
        """Current policy of the environment."""
        if self._policy is None:
            return self.base_model

        return self._policy

    @policy.setter
    def policy(self, policy: Policy[DataType]) -> None:
        """Set the current policy of the environment."""
        self._policy = policy

    @property
    def is_policy_set(self) -> bool:
        """Whether a custom policy has been set."""
        return self.policy is not None

    @property
    def additive_policy(self) -> Optional[Policy[DataType]]:
        """Current additive policy of the environment."""
        return self._additive_policy

    @additive_policy.setter
    def additive_policy(self, additive_policy: Optional[Policy[DataType]]) -> None:
        """Set the current additive policy of the environment."""
        self._additive_policy = additive_policy

    @property
    def is_additive_policy_set(self) -> bool:
        """Whether a custom policy has been set."""
        return self.additive_policy is not None

    @abstractmethod
    def drift(
        self, x: DataType, t: torch.Tensor, **kwargs: dict[str, Any]
    ) -> tuple[DataType, torch.Tensor]:
        """Compute the drift term of the environment's dynamics.

        Parameters
        ----------
        x : DataType
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        **kwargs : dict[str, Any]
            Keyword arguments to the model.

        Returns
        -------
        drift : T
            The drift term at state x and time t.

        running_cost : torch.Tensor, shape (n,)
            Running cost :math:`L(x_t, t)` of the policy for the given (state, timestep)-pair.
        """

    def diffusion(self, x: DataType, t: torch.Tensor) -> DataType:
        """Compute the diffusion term of the environment's dynamics.

        Parameters
        ----------
        x : DataType
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        Returns
        -------
        diffusion : DataType
            The diffusion term at time t.
        """
        return self.scheduler.sigma(t) * x.ones_like()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: Optional[torch.device] = None,
        **kwargs: dict[str, Any],
    ) -> tuple[list[DataType], torch.Tensor, torch.Tensor]:
        r"""Sample n trajectories from the environment.

        Parameters
        ----------
        n : int
            Number of trajectories to sample.

        device : torch.device, default "cpu"
            The device to perform computations on.

        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to the base model at every timestep (e.g. text
            embedding or class label).

        Returns
        -------
        trajectories : list of DataType, length discretization_steps
            The sampled trajectories, containing x_t.

        costs : torch.Tensor, shape (discretization_steps, n)
            The costs associated with each trajectory, i.e., :math:`c_t = \int_t^1 \| a_s(x_s, s) -
            \hat{a}_s(x_s, s) \|^2 ds - r(x_1)`.

        rewards : torch.Tensor, shape (n,)
            The final reward for each trajectory, i.e., :math:`r(x_1)`.

        """
        if device is None:
            device = torch.device("cpu")

        x = self.base_model.sample_p0(n)
        trajectories = [x]

        running_costs = torch.zeros(self.discretization_steps, n)

        t = torch.linspace(0, 1, self.discretization_steps + 1)
        for i, (t0, t1) in enumerate(pairwise(t)):
            dt = t1 - t0
            t_curr = t0 * torch.ones(n, device=device)

            # Discrete step of SDE
            drift, running_cost = self.drift(x, t_curr, **kwargs)
            diffusion = self.diffusion(x, t_curr)
            x += dt * drift + torch.sqrt(dt) * diffusion * x.randn_like()

            running_costs[i] = running_cost
            trajectories.append(x)

        x = self.base_model.postprocess(x)
        rewards = self.reward(x)
        running_costs = running_costs / self.discretization_steps
        running_costs = torch.cat([running_costs, -rewards], dim=1)
        # Reverse cumulative sum
        costs = running_costs.flip(0).cumsum(0).flip(0)
        return trajectories, costs, rewards
