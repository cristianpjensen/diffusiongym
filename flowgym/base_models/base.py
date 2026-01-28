"""Abstract base class for base models used in flow matching and diffusion."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Optional

import torch
from torch import nn

from flowgym.schedulers import Scheduler
from flowgym.types import D

OutputType = Literal["epsilon", "endpoint", "velocity", "score"]


class BaseModel(ABC, nn.Module, Generic[D]):
    """Abstract base class for base models used in flow matching and diffusion."""

    output_type: OutputType

    def __init__(self, device: Optional[torch.device]):
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device

    @property
    @abstractmethod
    def scheduler(self) -> Scheduler[D]:
        """Base model-dependent scheduler used for sampling."""

    @abstractmethod
    def sample_p0(self, n: int, **kwargs: Any) -> tuple[D, dict[str, Any]]:
        """Sample n data points from the base distribution p0.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        samples : D
            Samples from the base distribution p0.

        kwargs : dict
            Additional keyword arguments.
        """

    @abstractmethod
    def forward(self, x: D, t: torch.Tensor, **kwargs: Any) -> D:
        """Forward pass of the base model.

        Parameters
        ----------
        x : D
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        Returns
        -------
        output : D
            Output of the model.
        """

    def preprocess(self, x: D, **kwargs: Any) -> tuple[D, dict[str, Any]]:
        """Preprocess data and keyword arguments for the base model.

        Parameters
        ----------
        x : D
            Input data to preprocess.

        **kwargs : dict
            Additional keyword arguments to preprocess.

        Returns
        -------
        output : D
            Preprocessed data.

        kwargs : dict
            Preprocessed keyword arguments.
        """
        return x, kwargs

    def postprocess(self, x: D) -> D:
        """Postprocess samples x_1 (e.g., decode with VAE).

        Parameters
        ----------
        x : D
            Input data to postprocess.

        Returns
        -------
        output : D
            Postprocessed output.
        """
        return x

    def train_loss(
        self,
        x1: D,
        xt: Optional[D] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute loss for a single batch training step.

        Parameters
        ----------
        x1 : D
            Target data points.
        xt : Optional[D], default=None
            Noisy data points at time t. If None, will be sampled.
        t : Optional[torch.Tensor], shape (len(x1),), default=None
            Time steps. If None, will be sampled.

        **kwargs : dict
            Keyword arguments

        Returns
        -------
        loss : torch.Tensor, shape (len(x1),)
            Computed loss for the training step.
        """
        if t is None:
            t = torch.rand(len(x1), device=x1.device)

        assert t.shape == (len(x1),)

        alpha = self.scheduler.alpha(x1, t)
        beta = self.scheduler.beta(x1, t)

        if xt is None:
            x0 = x1.randn_like()
            xt = alpha * x1 + beta * x0
        else:
            assert len(xt) == len(x1)
            x0 = (xt - alpha * x1) / beta

        pred = self.forward(xt, t, **kwargs)

        target = None
        if self.output_type == "velocity":
            alpha_dot = self.scheduler.alpha_dot(x1, t)
            beta_dot = self.scheduler.beta_dot(x1, t)
            target = alpha_dot * x1 + beta_dot * x0
        elif self.output_type == "score":
            target = -x0 / beta
        elif self.output_type == "endpoint":
            target = x1
        elif self.output_type == "epsilon":
            target = x0
        else:
            raise ValueError(f"Unknown output type: {self.output_type}")

        return ((pred - target) ** 2).aggregate("mean")
