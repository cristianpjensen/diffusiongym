"""Public implementation of Value Matching."""

import logging
import os
import time
from typing import Callable, Optional

import polars as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from flowgym import D, Environment
from flowgym.utils import ValuePolicy
from value_matching.utils import Report


def value_matching(
    value_network: nn.Module,
    env: Environment[D],
    batch_size: int = 128,
    num_iterations: int = 1000,
    lr: float = 1e-4,
    log_every: Optional[int] = None,
    exp_dir: Optional[os.PathLike[str]] = None,
    fn_every: Optional[Callable[[int, Environment[D]], None]] = None,
    kwargs: Optional[dict] = None,
) -> None:
    """Run value matching to train a value network.

    Parameters
    ----------
    value_network : nn.Module
        The value function network, :math:`V(x, t)`.

    env : Environment
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

    fn_every : Optional[Callable[[int, Environment], None]], default=None
        A function to call every `log_every` iterations with the current iteration and environment.

    kwargs : Optional[dict], default=None
        Additional keyword arguments to pass to the value network.
    """
    value_network.to(env.device)

    opt = torch.optim.Adam(value_network.parameters(), lr=lr)
    warmup = LinearLR(opt, start_factor=1.0, total_iters=100)
    cosine = CosineAnnealingLR(opt, T_max=num_iterations - 100, eta_min=1e-2 * lr)
    scheduler = SequentialLR(opt, [warmup, cosine], milestones=[100])

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    if exp_dir is not None:
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    if kwargs is None:
        kwargs = {}

    weights = get_loss_weights(env)
    report = Report()

    # Set policy
    control = ValuePolicy(value_network, env.memoryless_schedule)
    env.control_policy = control

    for it in range(1, num_iterations + 1):
        with torch.no_grad():
            _, trajectories, _, _, running_costs, rewards, valids, costs, current_kwargs = (
                env.sample(
                    batch_size,
                    pbar=False,
                    **kwargs,
                )
            )

        opt.zero_grad()

        # Accumulate gradients
        total_loss = 0.0
        for idx, t in enumerate(torch.linspace(2e-2, 1, env.discretization_steps + 1)):
            x_t = trajectories[idx].to(env.device)
            t_curr = t.expand(batch_size).to(env.device)
            weight = weights[idx]

            output = value_network(x_t, t_curr, **current_kwargs).squeeze(-1)
            target = costs[idx].to(env.device)

            loss = (weight * ((output - target) / env.reward_scale).square()).mean()
            loss /= env.discretization_steps

            if loss.isnan().any() or loss.isinf().any():
                raise ValueError("Loss is NaN or Inf")

            total_loss += loss.item()
            loss.backward()  # type: ignore[no-untyped-call]

        grad_norm = nn.utils.clip_grad_norm_(value_network.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if exp_dir is not None:
            torch.save(value_network.state_dict(), os.path.join(exp_dir, "checkpoints", "last.pt"))

        report.update(
            loss=total_loss,
            r_mean=rewards.mean().item(),
            r_std=rewards.std().item(),
            valids=valids.float().mean().item(),
            running_cost=running_costs[:-1].mean().item(),
            cost=(costs[0] - costs[-1]).mean().item(),
            grad_norm=grad_norm.item(),
        )

        # Save stats
        if exp_dir is not None:
            row = {
                "iteration": it,
                "timestamp": int(time.time()),
                **{k: v[-1] for k, v in report.stats.items()},
            }
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
                    os.path.join(exp_dir, "checkpoints", f"iter_{it:06d}.pt"),
                )

        if fn_every is not None:
            fn_every(it, env)


def get_loss_weights(env: Environment[D]) -> torch.Tensor:
    """Compute loss weights for value matching, inversely proportional to future variance.

    Parameters
    ----------
    env : Environment
        The environment to compute the loss weights for.

    Returns
    -------
    weights : torch.Tensor, shape (discretization_steps + 1,)
        The loss weights for each time step.
    """
    ts = torch.linspace(2e-2, 1, env.discretization_steps + 1, device=env.device)
    dt = ts[1] - ts[0]
    # todo: fix access to scheduler
    sigmas = env.scheduler._sigma(ts).square().mean(dim=-1)
    cumsigmas = sigmas.flip(0).cumsum(0).flip(0) * dt
    weights: torch.Tensor = 1 / (1 + 0.5 * cumsigmas)
    return weights
