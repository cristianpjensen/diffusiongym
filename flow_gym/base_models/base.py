"""Abstract base class for base models used in flow matching and diffusion."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional

import torch
from torch import nn

from flow_gym.schedulers import Scheduler
from flow_gym.types import DataType


class BaseModel(ABC, nn.Module, Generic[DataType]):
    """Abstract base class for base models used in flow matching and diffusion."""

    def __init__(self, device: Optional[torch.device]):
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.device = device

    @property
    @abstractmethod
    def scheduler(self) -> Scheduler[DataType]:
        """Base model-dependent scheduler used for sampling."""

    @abstractmethod
    def sample_p0(self, n: int) -> DataType:
        """Sample n data points from the base distribution p0.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : DataType
            Samples from the base distribution p0.
        """

    @abstractmethod
    def forward(self, x: DataType, t: torch.Tensor, **kwargs: dict[str, Any]) -> DataType:
        """Forward pass of the base model.

        Parameters
        ----------
        x : DataType
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        Returns
        -------
        output : DataType
            Output of the model.
        """

    def postprocess(self, x: DataType) -> DataType:
        """Postprocess samples x_1 (e.g., decode with VAE).

        Parameters
        ----------
        x : DataType
            Input data to postprocess.

        Returns
        -------
        output : DataType
            Postprocessed output.
        """
        return x
