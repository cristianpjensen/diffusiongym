"""Base environment classes and interfaces for flowgym."""

from abc import ABC, abstractmethod
from itertools import pairwise
from typing import Any, Generic, Iterable, Optional, Protocol

import torch
from tqdm.auto import tqdm

from flowgym.base_models import BaseModel
from flowgym.rewards import Reward
from flowgym.schedulers import MemorylessNoiseSchedule, Scheduler
from flowgym.types import D


class Policy(Protocol[D]):
    """General protocol for a policy function."""

    def __call__(self, x: D, t: torch.Tensor, **kwargs: Any) -> D: ...


class Environment(ABC, Generic[D]):
    """Abstract base class for all environments.

    Parameters
    ----------
    base_model : BaseModel[D]
        The base generative model used in the environment.

    reward : Reward[D]
        The reward function used to compute the final reward.

    discretization_steps : int
        The number of discretization steps to use when sampling trajectories.
    """

    def __init__(
        self,
        base_model: BaseModel[D],
        reward: Reward[D],
        discretization_steps: int,
        reward_scale: float = 1.0,
    ):
        self.base_model = base_model
        self.reward = reward
        self.discretization_steps = discretization_steps
        self.reward_scale = reward_scale
        self._policy: Optional[Policy[D]] = None
        self._control_policy: Optional[Policy[D]] = None
        self.memoryless_schedule = MemorylessNoiseSchedule(self.scheduler)

    @property
    def device(self) -> torch.device:
        """Get the device of the base model."""
        return self.base_model.device

    @property
    def scheduler(self) -> Scheduler[D]:
        """Get the scheduler of the base model."""
        return self.base_model.scheduler

    @property
    def policy(self) -> Policy[D]:
        """Current policy (replacement of base model) of the environment."""
        if self._policy is None:
            return self.base_model

        return self._policy

    @policy.setter
    def policy(self, policy: Policy[D]) -> None:
        """Set the current policy of the environment."""
        self._policy = policy

    @property
    def is_policy_set(self) -> bool:
        """Whether a custom policy has been set."""
        return self._policy is not None

    @property
    def control_policy(self) -> Optional[Policy[D]]:
        """Current control policy u(x, t) of the environment."""
        return self._control_policy

    @control_policy.setter
    def control_policy(self, control_policy: Optional[Policy[D]]) -> None:
        """Set the current control policy of the environment."""
        self._control_policy = control_policy

    @property
    def is_control_policy_set(self) -> bool:
        """Whether a custom policy has been set."""
        return self.control_policy is not None

    @abstractmethod
    def pred_final(
        self,
        x: D,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> D:
        """Compute the final state prediction from the current state.

        Parameters
        ----------
        x : D
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        **kwargs : dict
            Keyword arguments to the model.

        Returns
        -------
        final : D
            The predicted final state from state x and time t.
        """

    @abstractmethod
    def drift(
        self,
        x: D,
        t: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[D, torch.Tensor]:
        """Compute the drift term of the environment's dynamics.

        Parameters
        ----------
        x : D
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        **kwargs : dict
            Keyword arguments to the model.

        Returns
        -------
        drift : D
            The drift term at state x and time t.

        running_cost : torch.Tensor, shape (n,)
            Running cost :math:`L(x_t, t)` of the policy for the given (state, timestep)-pair.
        """

    def diffusion(self, x: D, t: torch.Tensor) -> D:
        """Compute the diffusion term of the environment's dynamics.

        Parameters
        ----------
        x : D
            The current state.

        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        Returns
        -------
        diffusion : D
            The diffusion term at time t.
        """
        return self.scheduler.sigma(x, t)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        pbar: bool = True,
        x0: Optional[D] = None,
        **kwargs: Any,
    ) -> tuple[
        D,
        list[D],
        list[D],
        list[D],
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

        x0 : D, optional
            Initial states to start the trajectories from. If None, samples from :math:`p_0`.

        **kwargs : dict
            Additional keyword arguments to pass to the base model at every timestep (e.g. text
            embedding or class label).

        Returns
        -------
        sample : D
            The final states :math:`x_1` of the sampled trajectory.

        trajectories : list of D, length discretization_steps+1
            The sampled trajectories, containing x_t.

        drifts : list of D, length discretization_steps
            The drift terms at each timestep.

        noises : list of D, length discretization_steps
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

        # Set initial state if provided
        if x0 is not None:
            x = x0.to(self.base_model.device)

        x, kwargs = self.base_model.preprocess(x, **kwargs)

        trajectories = [x.to("cpu")]
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
            trajectories.append(x.detach().to("cpu"))
            drifts.append(drift.detach().to("cpu"))
            noises.append(epsilon.detach().to("cpu"))

        sample = self.base_model.postprocess(x)

        if self.reward.latent_space:
            rewards, valids = self.reward(x, **kwargs)
        else:
            rewards, valids = self.reward(sample, **kwargs)

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
        return sample, trajectories, drifts, noises, running_costs, rewards, valids, costs, kwargs
