"""Abstract base class for base models used in flow matching and diffusion."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, Optional

import torch
from torch import nn

from flow_gym.schedulers import Scheduler
from flow_gym.types import DataType

OutputType = Literal["epsilon", "endpoint", "velocity", "score"]


class BaseModel(ABC, nn.Module, Generic[DataType]):
    """Abstract base class for base models used in flow matching and diffusion."""

    output_type: OutputType

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
    def forward(self, x: DataType, t: torch.Tensor, **kwargs: Any) -> DataType:
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

    def preprocess(self, x: DataType, **kwargs: Any) -> tuple[DataType, dict[str, Any]]:
        """Preprocess data and keyword arguments for the base model.

        Parameters
        ----------
        x : DataType
            Input data to preprocess.

        **kwargs : dict
            Additional keyword arguments to preprocess.

        Returns
        -------
        output : DataType
            Preprocessed data.

        kwargs : dict
            Preprocessed keyword arguments.
        """
        return x, kwargs

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
