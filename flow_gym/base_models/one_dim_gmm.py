"""Base model for 1D Gaussian mixture model (GMM)."""

from typing import Any, Optional, cast

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn
from tqdm.auto import trange

from flow_gym.registry import base_model_registry
from flow_gym.schedulers import OptimalTransportScheduler, Scheduler
from flow_gym.types import FGTensor

from .base import BaseModel


@base_model_registry.register("1d/gmm")
class OneDimensionalBaseModel(BaseModel[FGTensor]):
    """Base model for 1D Gaussian mixture model (GMM).

    Keep in mind that this trains the model, so it may take a minute to load.
    """

    output_type = "velocity"

    def __init__(
        self, device: Optional[torch.device], scheduler: Optional[Scheduler[FGTensor]] = None
    ):
        super().__init__(device)

        if device is None:
            device = torch.device("cpu")

        self.device = device

        if scheduler is None:
            scheduler = OptimalTransportScheduler()

        self._scheduler = scheduler
        self.model = train_1d_gaussian(scheduler, device).to(device)

    @property
    def scheduler(self) -> Scheduler[FGTensor]:
        """Optimal transport scheduler."""
        return self._scheduler

    def sample_p0(self, n: int) -> FGTensor:
        """Sample n data points from the base distribution p0.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : FGTensor, shape (n, 1)
            Samples from the base distribution p0.
        """
        return FGTensor(torch.randn(n, 1, device=self.device))

    def forward(self, x: FGTensor, t: torch.Tensor, **kwargs: Any) -> FGTensor:
        """Forward pass of the base model.

        Parameters
        ----------
        x : FGTensor, shape (n, 1)
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        Returns
        -------
        output : DataType
            Output of the model.
        """
        out = cast("torch.Tensor", self.model(x, t))
        return FGTensor(out)


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) with 1 layer that appends time as an additional input.

    Parameters
    ----------
    input_dim : int
        Input dimension.

    hidden_dim : int
        Hidden dimension.

    output_dim : int
        Output dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

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
        t = t.unsqueeze(-1)
        x_and_t = torch.cat([x, t], dim=-1)
        out = cast("torch.Tensor", self.net(x_and_t))
        return out


def train_1d_gaussian(scheduler: Scheduler[FGTensor], device: torch.device) -> nn.Module:
    """Trains a one-dimensional Gaussian mixture model.

    Takes about 20 seconds on an RTX 4090.

    Parameters
    ----------
    scheduler : Scheduler[FGTensor]
        Scheduler to use for training.

    device : torch.device
        Device to use for training.

    Returns
    -------
    model : nn.Module
        Trained model.
    """
    p1 = dist.MixtureSameFamily(  # type: ignore
        dist.Categorical(torch.ones(2)),  # type: ignore
        dist.Normal(torch.Tensor([0.0, 3.0]), torch.Tensor([1.0, 0.4])),  # type: ignore
    )

    mlp = MLP(1, 64, 1).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    mlp.train()
    for _ in trange(10_000, desc="training 1d gmm model"):
        x1 = FGTensor(p1.sample((4096, 1)).to(device))  # type: ignore
        x0 = x1.randn_like()
        t = torch.rand(x1.shape[0], device=device)

        alpha = scheduler.alpha(x1, t)
        beta = scheduler.beta(x1, t)
        alpha_dot = scheduler.alpha_dot(x1, t)
        beta_dot = scheduler.beta_dot(x1, t)

        # Compute conditional flow and velocity
        xt = alpha * x1 + beta * x0
        dxt = alpha_dot * x1 + beta_dot * x0

        # Predict velocity and compute loss
        velocity_pred = mlp(xt, t)
        loss = F.mse_loss(velocity_pred, dxt)

        # Update model parameters
        opt.zero_grad()
        loss.backward()  # type: ignore
        opt.step()

    mlp.eval()
    return mlp
