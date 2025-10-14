"""Public implementation of Value Matching."""

import logging
import os
from typing import Any, Callable, Generic, Optional

import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from flow_gym.environments import BaseEnvironment
from flow_gym.schedulers import NoiseSchedule
from flow_gym.types import DataType
from flow_gym.utils import Report


def value_matching(
    value_network: nn.Module,
    env: BaseEnvironment[DataType],
    batch_size: int = 128,
    num_iterations: int = 1000,
    lr: float = 1e-4,
    log_every: Optional[int] = None,
    exp_dir: Optional[os.PathLike[str]] = None,
    fn_every: Optional[Callable[[int, BaseEnvironment[DataType]], None]] = None,
) -> None:
    """Run value matching to train a value network.

    Parameters
    ----------
    value_network : nn.Module
        The value function network, :math:`V(x, t)`.

    env : BaseEnvironment
        The environment to train the value function in.

    batch_size : int, default=128
        The batch size to use for training.

    num_iterations : int, default=1000
        The number of training iterations.

    lr : float, default=1e-4
        The learning rate for the optimizer.

    log_every : Optional[int], default=None
        How often to log training statistics. If None, it will log 100 times during training

    exp_dir : Optional[os.PathLike], default=None
        Directory to save training statistics and model checkpoints. If None, no files are saved.

    fn_every : Optional[Callable[[int, BaseEnvironment], None]], default=None
        A function to call every `log_every` iterations with the current iteration and environment.
    """
    value_network.to(env.device)
    opt = torch.optim.Adam(value_network.parameters(), lr=lr)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    if exp_dir is not None:
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    weights = get_loss_weights(env)
    report = Report()

    # Set policy
    control = ValuePolicy(value_network, env.memoryless_schedule)
    env.control_policy = control

    for it in range(1, num_iterations + 1):
        with torch.no_grad():
            _, trajectories, running_costs, rewards, costs, kwargs = env.sample(
                batch_size,
                pbar=False,
            )

        opt.zero_grad()

        # Accumulate gradients
        total_loss = 0.0
        for idx, t in enumerate(
            torch.linspace(1 / env.discretization_steps, 1, env.discretization_steps + 1)
        ):
            x_t = trajectories[idx].to_device(env.device)
            t_curr = t.expand(batch_size).to(env.device)
            pred = value_network(x_t, t_curr, **kwargs).squeeze(-1) / env.reward_scale
            target = costs[idx].to(env.device) / env.reward_scale

            loss = F.mse_loss(pred, target, reduction="mean")
            loss *= weights[idx] / env.discretization_steps

            if loss.isnan().any() or loss.isinf().any():
                raise ValueError("Loss is NaN or Inf")

            total_loss += loss.item()
            loss.backward()  # type: ignore[no-untyped-call]

        grad_norm = nn.utils.clip_grad_norm_(value_network.parameters(), 1.0)
        opt.step()

        if exp_dir is not None:
            torch.save(value_network.state_dict(), os.path.join(exp_dir, "checkpoints", "last.pt"))

        report.update(
            loss=total_loss,
            r_mean=rewards.mean().item(),
            r_std=rewards.std().item(),
            running_cost=running_costs[:-1].mean().item(),
            grad_norm=grad_norm.item(),
        )

        # Save stats
        if exp_dir is not None:
            row = {"iteration": it, **{k: v[-1] for k, v in report.stats.items()}}
            df = pl.DataFrame([row])
            stats_file = os.path.join(exp_dir, "training_stats.csv")
            write_header = not os.path.exists(stats_file)
            # Stream-write to stats_file with Polars
            with open(stats_file, "a", newline="") as f:
                df.write_csv(f, include_header=write_header)

        # Log stats and save weights
        if it % log_every == 0:
            logging.info(
                f"(step={it:06d}) {report}, "
                f"max vram={torch.cuda.max_memory_allocated() * 1e-9:.2f}GB"
            )

            if exp_dir is not None:
                torch.save(
                    value_network.state_dict(),
                    os.path.join(exp_dir, "checkpoints", f"iter_{it + 1:06d}.pt"),
                )

        if fn_every is not None:
            fn_every(it, env)


class ValuePolicy(nn.Module, Generic[DataType]):
    r"""Policy based on a value function, :math:`u(x, t) = -\sigma(t) \nabla_x V(x, t)`.

    Parameters
    ----------
    value_network : nn.Module
        The value function network, :math:`V(x, t)`.

    noise_schedule : NoiseSchedule
        The noise schedule, :math:`\sigma(t)`.
    """

    def __init__(self, value_network: nn.Module, noise_schedule: NoiseSchedule[DataType]) -> None:
        super().__init__()
        self.value_network = value_network
        self.noise_schedule = noise_schedule

    @torch.enable_grad()  # type: ignore[no-untyped-call]
    def forward(self, x: DataType, t: torch.Tensor, **kwargs: Any) -> DataType:
        """Compute control action based on value function gradient."""
        x = x.with_requires_grad()
        value_pred = self.value_network(x, t, **kwargs)
        sigma = self.noise_schedule(x, t)
        control: DataType = -sigma * x.gradient(value_pred)
        return control


def get_loss_weights(env: BaseEnvironment[DataType]) -> torch.Tensor:
    """Compute loss weights for value matching, inversely proportional to future variance.

    Parameters
    ----------
    env : BaseEnvironment
        The environment to compute the loss weights for.

    Returns
    -------
    weights : torch.Tensor, shape (discretization_steps + 1,)
        The loss weights for each time step.
    """
    ts = torch.linspace(0, 1, env.discretization_steps + 1, device=env.device)
    dt = ts[1] - ts[0]
    sigmas = env.scheduler._sigma(ts).square().mean(dim=-1)
    cumsigmas = sigmas.flip(0).cumsum(0).flip(0) * dt
    weights: torch.Tensor = 1 / (1 + 0.5 * cumsigmas)
    return weights
