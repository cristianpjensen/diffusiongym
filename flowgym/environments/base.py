"""Base environment classes and interfaces for flowgym."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Generic, Iterable, Optional, Protocol

import torch
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from flowgym.base_models import BaseModel
from flowgym.rewards import Reward
from flowgym.schedulers import MemorylessNoiseSchedule, Scheduler
from flowgym.types import D
from flowgym.utils import index_dict


@dataclass
class Sample(Generic[D]):
    sample: D
    latent: D
    trajectory: list[D]
    drifts: list[D]
    noises: list[D]
    running_costs: torch.Tensor
    rewards: torch.Tensor
    valids: torch.Tensor
    cost_functionals: torch.Tensor
    kwargs: dict[str, Any]

    def __post_init__(self):
        n = len(self.sample)
        assert len(self.latent) == n, f"latent batch size != sample batch size, got {len(self.latent)} != {n}"
        assert len(self.trajectory[0]) == n, f"trajectory batch size != sample batch size, got {len(self.trajectory[0])} != {n}"
        assert len(self.drifts[0]) == n, f"drift batch size != sample batch size, got {len(self.drifts[0])} != {n}"
        assert len(self.noises[0]) == n, f"noise batch size != sample batch size, got {len(self.noises[0])} != {n}"
        assert self.running_costs.shape[1] == n, f"running_costs batch size != sample batch size, got {self.running_costs.shape[0]} != {n}"
        assert self.rewards.shape[0] == n, f"rewards batch size != sample batch size, got {self.rewards.shape[0]} != {n}"
        assert self.valids.shape[0] == n, f"valids batch size != sample batch size, got {self.valids.shape[0]} != {n}"
        assert self.cost_functionals.shape[1] == n, f"cost functionals batch size != sample batch size, got {self.cost_functionals.shape[0]} != {n}"

        num_steps = len(self.trajectory) - 1
        assert len(self.drifts) == num_steps, f"drifts length != number of steps, got {len(self.drifts)} != {num_steps}"
        assert len(self.noises) == num_steps, f"noises length != number of steps, got {len(self.noises)} != {num_steps}"
        assert self.running_costs.shape[0] == num_steps, f"running_costs length != number of steps, got {self.running_costs.shape[0]} != {num_steps}"
        assert self.cost_functionals.shape[0] == num_steps + 1, (
            f"cost_functionals length != number of steps + 1, got {self.cost_functionals.shape[0]} != {num_steps + 1}"
        )

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx: int) -> "Sample[D]":
        return Sample(
            sample=self.sample[idx],
            latent=self.latent[idx],
            trajectory=[state[idx] for state in self.trajectory],
            drifts=[drift[idx] for drift in self.drifts],
            noises=[noise[idx] for noise in self.noises],
            running_costs=self.running_costs[:, idx : idx + 1],
            rewards=self.rewards[idx : idx + 1],
            valids=self.valids[idx : idx + 1],
            cost_functionals=self.cost_functionals[:, idx : idx + 1],
            kwargs=index_dict(self.kwargs, idx, idx + 1),
        )

    @staticmethod
    def concat(samples: list["Sample[D]"]) -> "Sample[D]":
        data_type = type(samples[0].sample)
        num_steps = len(samples[0].trajectory) - 1

        all_kwargs = []
        for sample in samples:
            for i in range(len(sample)):
                all_kwargs.append(index_dict(sample.kwargs, i))

        return Sample(
            sample=data_type.collate([x.sample for x in samples]),
            latent=data_type.collate([x.latent for x in samples]),
            trajectory=[data_type.collate([x.trajectory[t] for x in samples]) for t in range(num_steps + 1)],
            drifts=[data_type.collate([x.drifts[t] for x in samples]) for t in range(num_steps)],
            noises=[data_type.collate([x.noises[t] for x in samples]) for t in range(num_steps)],
            running_costs=torch.cat([x.running_costs for x in samples], dim=1),
            rewards=torch.cat([x.rewards for x in samples], dim=0),
            valids=torch.cat([x.valids for x in samples], dim=0),
            cost_functionals=torch.cat([x.cost_functionals for x in samples], dim=1),
            kwargs=default_collate(all_kwargs),  # type: ignore
        )


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
    ) -> Sample[D]:
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
        Sample[D]
            A Sample object containing the sampled trajectories and associated data.
        """
        x, kwargs = self.base_model.sample_p0(n, **kwargs)

        # Set initial state if provided
        if x0 is not None:
            x = x0.to(self.base_model.device)

        x, kwargs = self.base_model.preprocess(x, **kwargs)

        trajectory = [x.to("cpu")]
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
            trajectory.append(x.detach().to("cpu"))
            drifts.append(drift.detach().to("cpu"))
            noises.append(epsilon.detach().to("cpu"))

        sample = self.base_model.postprocess(x)
        rewards, valids = self.reward(sample, x, **kwargs)

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
        return Sample(sample.to("cpu"), x.to("cpu"), trajectory, drifts, noises, running_costs, rewards, valids, costs, kwargs)

    def batch_sample(self, n: int, batch_size: int, **kwargs: Any) -> Sample[D]:
        """Sample n trajectories from the environment in batches.

        Parameters
        ----------
        n : int
            Number of trajectories to sample.
        batch_size : int
            Batch size for sampling.
        **kwargs : dict
            Additional keyword arguments to pass to the base model at every timestep (e.g. text
            embedding or class label).
        """
        samples: list[Sample[D]] = []

        for i in range(0, n, batch_size):
            current_n = min(batch_size, n - i)
            current_kwargs = index_dict(kwargs, i, i + current_n)
            samples.append(self.sample(current_n, pbar=False, **current_kwargs))

        return Sample.concat(samples)
