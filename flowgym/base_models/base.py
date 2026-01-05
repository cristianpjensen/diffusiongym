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
