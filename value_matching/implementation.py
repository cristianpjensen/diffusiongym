"""Public implementation of Value Matching."""

import logging
import time
from pathlib import Path
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
    exp_dir: Optional[Path | str] = None,
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

    log_every : int, default=100 times
        How often to log training statistics.

    exp_dir : Path | str, default=no saved files
        Directory to save training statistics and model checkpoints.

    fn_every : (int, Environment) -> None, default=no logging
        A function to call every `log_every` iterations with the current iteration and environment.

    kwargs : dict, default={}
        Additional keyword arguments to pass to the value network.
    """
    value_network.to(env.device)

    opt = torch.optim.Adam(value_network.parameters(), lr=lr)
    warmup = LinearLR(opt, start_factor=1.0, total_iters=100)
    cosine = CosineAnnealingLR(opt, T_max=num_iterations - 100, eta_min=1e-2 * lr)
    scheduler = SequentialLR(opt, [warmup, cosine], milestones=[100])

    if isinstance(exp_dir, str):
        exp_dir = Path(exp_dir)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)

    if kwargs is None:
        kwargs = {}

    weights = get_loss_weights(env)
    report = Report()

    # Set policy
    control = ValuePolicy(value_network, env.memoryless_schedule)
    env.control_policy = control

    for it in range(1, num_iterations + 1):
        with torch.no_grad():
            out = env.sample(batch_size, pbar=False, **kwargs)
            _, trajectories, _, _, running_costs, rewards, valids, costs, current_kwargs = out

        opt.zero_grad()

        # Accumulate gradients
        total_loss = 0.0
        for idx, t in enumerate(torch.linspace(2e-2, 1, env.discretization_steps + 1)):
            x_t = trajectories[idx].to(env.device)
            t_curr = t.expand(batch_size).to(env.device)
            weight = weights[idx]

            output = value_network(x_t, t_curr, **current_kwargs).squeeze(-1)
            target = costs[idx].to(env.device)

            loss = (weight * (output - target).square()).mean()
            loss /= env.discretization_steps

            if loss.isnan().any() or loss.isinf().any():
                raise ValueError("Loss is NaN or Inf")

            total_loss += loss.item()
            loss.backward()  # type: ignore[no-untyped-call]

        grad_norm = nn.utils.clip_grad_norm_(value_network.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if exp_dir is not None:
            torch.save(value_network.state_dict(), exp_dir / "checkpoints" / "last.pt")

        report.update(
            loss=total_loss,
            r_mean=rewards[valids].mean().item(),
            r_std=rewards[valids].std().item(),
            validity=valids.float().mean().item(),
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
            stats_file = exp_dir / "training_stats.csv"
            # Stream-write to stats_file with Polars
            with open(stats_file, "a", newline="") as f:
                df.write_csv(f, include_header=not stats_file.exists())

        # Log stats and save weights
        if it % log_every == 0:
            logging.info(f"(step={it:06d}) {report}, vram={torch.cuda.max_memory_allocated() * 1e-9:.2f}GB")

            if exp_dir is not None:
                torch.save(value_network.state_dict(), exp_dir / "checkpoints" / f"iter_{it:06d}.pt")

        if fn_every is not None:
            fn_every(it, env)


@torch.no_grad()
def get_loss_weights(env: Environment[D]) -> torch.Tensor:
    """Compute loss weights for value matching, inversely proportional to future variance.

    Parameters
    ----------
    env : Environment[D]
        The environment to compute the loss weights for.

    Returns
    -------
    weights : torch.Tensor, shape (discretization_steps + 1,)
        The loss weights for each time step.
    """
    # Get a single sample so we can compute sigma
    x = env.base_model.sample_p0(1)[0]
    xs = type(x).collate([x] * (env.discretization_steps + 1)).to(env.device)

    ts = torch.linspace(2e-2, 1, env.discretization_steps + 1, device=env.device)
    dt = ts[1] - ts[0]

    sigmas = (env.scheduler.sigma(xs, ts) ** 2).aggregate(reduction="mean")
    rev_cum_sigmas = sigmas.flip(0).cumsum(0).flip(0) * dt
    return 1 / (env.reward_scale * (1 + 0.5 * rev_cum_sigmas))
