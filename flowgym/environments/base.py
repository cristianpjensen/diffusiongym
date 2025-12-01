"""Base environment classes and interfaces for flow_gym."""

from abc import ABC, abstractmethod
from itertools import pairwise
from typing import Any, Generic, Iterable, Optional, Protocol

import torch
from tqdm.auto import tqdm

from flowgym.base_models import BaseModel
from flowgym.rewards import Reward
from flowgym.schedulers import MemorylessNoiseSchedule, Scheduler
from flowgym.types import DataType


class Policy(Protocol[DataType]):
    """General protocol for a policy function."""

    def __call__(self, x: DataType, t: torch.Tensor, **kwargs: Any) -> DataType: ...  # noqa: D102


class Environment(ABC, Generic[DataType]):
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
        reward_scale: float = 1.0,
    ):
        self.base_model = base_model
        self.reward = reward
        self.discretization_steps = discretization_steps
        self.reward_scale = reward_scale
        self._policy: Optional[Policy[DataType]] = None
        self._control_policy: Optional[Policy[DataType]] = None
        self.memoryless_schedule = MemorylessNoiseSchedule(self.scheduler)

    @property
    def device(self) -> torch.device:
        """Get the device of the base model."""
        return self.base_model.device

    @property
    def scheduler(self) -> Scheduler[DataType]:
        """Get the scheduler of the base model."""
        return self.base_model.scheduler

    @property
    def policy(self) -> Policy[DataType]:
        """Current policy (replacement of base model) of the environment."""
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
        return self._policy is not None

    @property
    def control_policy(self) -> Optional[Policy[DataType]]:
        """Current control policy u(x, t) of the environment."""
        return self._control_policy

    @control_policy.setter
    def control_policy(self, control_policy: Optional[Policy[DataType]]) -> None:
        """Set the current control policy of the environment."""
        self._control_policy = control_policy

    @property
    def is_control_policy_set(self) -> bool:
        """Whether a custom policy has been set."""
        return self.control_policy is not None

    @abstractmethod
    def pred_final(
        self,
        x: DataType,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> DataType:
        """Compute the final state prediction from the current state.

        Parameters
        ----------
        x : DataType
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        **kwargs : dict
            Keyword arguments to the model.

        Returns
        -------
        final : DataType
            The predicted final state from state x and time t.
        """

    @abstractmethod
    def drift(
        self,
        x: DataType,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[DataType, torch.Tensor]:
        """Compute the drift term of the environment's dynamics.

        Parameters
        ----------
        x : DataType
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        **kwargs : dict
            Keyword arguments to the model.

        Returns
        -------
        drift : DataType
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
        return self.scheduler.sigma(x, t)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        pbar: bool = True,
        **kwargs: Any,
    ) -> tuple[
        DataType,
        list[DataType],
        list[DataType],
        list[DataType],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        r"""Sample n trajectories from the environment.

        Parameters
        ----------
        n : int
            Number of trajectories to sample.

        pbar : bool, default: True
            Whether to display a progress bar.

        **kwargs : dict
            Additional keyword arguments to pass to the base model at every timestep (e.g. text
            embedding or class label).

        Returns
        -------
        sample : DataType
            The final states :math:`x_1` of the sampled trajectory.

        trajectories : list of DataType, length discretization_steps+1
            The sampled trajectories, containing x_t.

        drifts : list of DataType, length discretization_steps
            The drift terms at each timestep.

        noises : list of DataType, length discretization_steps
            The noise terms at each timestep.

        running_costs : torch.Tensor, shape (discretization_steps, n)
            The running costs :math:`L(x_t, t)` of the policy at each timestep.

        rewards : torch.Tensor, shape (n,)
            The final reward for each trajectory, i.e., :math:`r(x_1)`.

        valids : torch.Tensor, shape (n,)
            The validity indicators for each trajectory (1 if valid, 0 if invalid).

        costs : torch.Tensor, shape (discretization_steps, n)
            The costs associated with each trajectory, i.e., :math:`c_t = \int_t^1 \| a_s(x_s, s) -
            \hat{a}_s(x_s, s) \|^2 ds - r(x_1)`.

        kwargs : dict[str, Any]
            Additional keyword arguments passed to the base model at every timestep.
        """
        x, kwargs = self.base_model.sample_p0(n, **kwargs)
        x, kwargs = self.base_model.preprocess(x, **kwargs)

        trajectories = [x.to_device("cpu")]
        drifts = []
        noises = []
        running_costs = torch.zeros(self.discretization_steps, n)

        # Start at a very small number, instead of 0, to avoid singularities
        t = torch.linspace(2e-2, 1, self.discretization_steps + 1)
        iterator: Iterable[tuple[int, tuple[Any, Any]]] = enumerate(pairwise(t))
        if pbar:
            iterator = tqdm(iterator, total=self.discretization_steps)

        for i, (t0, t1) in iterator:
            dt = t1 - t0
            t_curr = t0 * torch.ones(n, device=self.base_model.device)

            # Discrete step of SDE
            drift, running_cost = self.drift(x, t_curr, **kwargs)
            diffusion = self.diffusion(x, t_curr)
            epsilon = x.randn_like()
            x += dt * drift + torch.sqrt(dt) * diffusion * epsilon

            running_costs[i] = running_cost
            trajectories.append(x.to_device("cpu"))
            drifts.append(drift.to_device("cpu"))
            noises.append(epsilon.to_device("cpu"))

        x = self.base_model.postprocess(x)

        rewards, valids = self.reward(x, **kwargs)
        rewards = rewards.cpu()
        valids = valids.cpu()
        costs = torch.cat(
            [
                running_costs / self.discretization_steps,
                -self.reward_scale * rewards.unsqueeze(0),
            ],
            dim=0,
        )
        # Reverse cumulative sum
        costs = costs.flip(0).cumsum(0).flip(0)
        return x, trajectories, drifts, noises, running_costs, rewards, valids, costs, kwargs
