"""Base model for 1D Gaussian mixture model (GMM)."""

import math
from typing import Any, Optional

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn

from flowgym.registry import base_model_registry
from flowgym.schedulers import OptimalTransportScheduler, Scheduler
from flowgym.types import FlowTensor
from flowgym.utils import append_dims, train_base_model

from .base import BaseModel


@base_model_registry.register("1d/gmm")
class OneDimensionalBaseModel(BaseModel[FlowTensor]):
    """Base model for 1D Gaussian mixture model (GMM).

    Keep in mind that this trains the model, so it may take a minute to load.
    """

    output_type = "velocity"

    def __init__(
        self,
        device: Optional[torch.device],
        scheduler: Optional[Scheduler[FlowTensor]] = None,
    ):
        super().__init__(device)

        if device is None:
            device = torch.device("cpu")

        self.device = device

        if scheduler is None:
            scheduler = OptimalTransportScheduler()

        self._scheduler = scheduler

        p1 = dist.MixtureSameFamily(  # type: ignore
            dist.Categorical(torch.ones(2)),  # type: ignore
            dist.Normal(torch.Tensor([0.0, 3.0]), torch.Tensor([1.0, 0.4])),  # type: ignore
        )
        data = [FlowTensor(p1.sample((4096, 1)).to(device))]  # type: ignore

        mlp = MLP(1, 1).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        train_base_model(self, data, epochs=1_000, batch_size=512, opt=opt, pbar=True)

    @property
    def scheduler(self) -> Scheduler[FlowTensor]:
        """Optimal transport scheduler."""
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        """Sample n data points from the base distribution p0.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        kwargs : dict
            Additional keyword arguments (unused).

        Returns
        -------
        samples : FlowTensor, shape (n, 1)
            Samples from the base distribution p0.

        kwargs : dict
            Additional keyword arguments (unused).
        """
        return FlowTensor(torch.randn(n, 1, device=self.device)), kwargs

    def forward(self, x: FlowTensor, t: torch.Tensor, **kwargs: Any) -> FlowTensor:
        """Forward pass of the base model.

        Parameters
        ----------
        x : FlowTensor, shape (n, 1)
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        Returns
        -------
        output : FlowTensor
            Output of the model.
        """
        return FlowTensor(self.model(x.data, t))


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_dim: int = 128,
        cond_dim: int = 256,
        depth: int = 3,
        width: int = 256,
        window_size: float = 1000.0,
        t_mult: float = 1000.0,
    ) -> None:
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim, cond_dim, window_size, t_mult)

        blocks = [Block(in_dim, width, cond_dim)]
        for _ in range(depth - 1):
            blocks.append(Block(width, width, cond_dim))

        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(width, out_dim, bias=True)

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Parameters
        ----------
        x : torch.Tensor, shape (n, input_dim)
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        Returns
        -------
        output : torch.Tensor, shape (n, output_dim)
            Output of the MLP.
        """
        cond = self.time_embed(t)
        x_ = x.flatten(start_dim=t.ndim)
        for block in self.blocks:
            x_ = block(x_, cond)

        return self.head(x_).reshape(*x.shape[:-1], -1)


class Block(nn.Module):
    """A single block of the MLP."""

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.film = FiLM(out_dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass of the block."""
        return F.silu(self.film(self.norm(self.linear(x)), cond))


class SinusoidalTimeEmbedding(nn.Module):
    """t -> R^d sinusoidal embedding + MLP."""

    def __init__(
        self,
        dim: int = 128,
        hidden_dim: int = 256,
        window_size: float = 1000.0,
        t_mult: float = 1000.0,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

        self.t_mult = t_mult
        half = dim // 2
        freqs = torch.exp(-math.log(window_size) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal time embedding."""
        # Assuming t is in [0, 1], scale to [0, t_mult] as is usual for diffusion models
        t = t.float() * self.t_mult
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(-2)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        result: torch.Tensor = self.mlp(emb)
        return result


class FiLM(nn.Module):
    """Feature-wise linear modulation h -> gamma(t) * h + beta(t)."""

    def __init__(self, n_channels: int, cond_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * n_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation."""
        gamma, beta = self.to_scale_shift(cond).chunk(2, dim=-1)
        gamma = append_dims(gamma, x.ndim)
        beta = append_dims(beta, x.ndim)
        return x * (1 + gamma) + beta
