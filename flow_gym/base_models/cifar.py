"""Pre-trained base model for CIFAR-10."""

from typing import TYPE_CHECKING, Any, cast

import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from flow_gym.schedulers import DiffusionScheduler

from .base import BaseModel

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d import UNet2DModel


class CIFARBaseModel(BaseModel[torch.Tensor]):
    """Pre-trained diffusion model on CIFAR-10 32x32.

    Uses the `google/ddpm-cifar10-32` model from the `diffusers` library.

    Examples
    --------
    >>> device = torch.device("cpu")
    >>> model = CIFARBaseModel().to(device)
    >>> x = model.sample_p0(8).to(device)
    >>> t = torch.rand(8, device=device)
    >>> output = model(x, t)
    >>> output.shape
    torch.Size([8, 3, 32, 32])
    """

    def __init__(self) -> None:
        super().__init__()

        pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
        self.unet: UNet2DModel = pipe.unet
        self._scheduler = DiffusionScheduler(pipe.scheduler.alphas_cumprod)

    @property
    def scheduler(self) -> DiffusionScheduler:
        """Scheduler used for sampling."""
        return self._scheduler

    def sample_p0(self, n: int) -> torch.Tensor:
        """Sample n datapoints from the base distribution p0.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : torch.Tensor, shape (n, 3, 32, 32)
            Samples from the base distribution p0.

        Notes
        -----
        The base distribution p0 is a standard Gaussian distribution.
        """
        return torch.randn(n, 3, 32, 32)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        r"""Forward pass of the model, outputting :math:`\epsilon(x_t, t)`.

        Parameters
        ----------
        x : torch.Tensor, shape (n, 3, 32, 32)
            Input data.
        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].
        **kwargs : dict
            Additional keyword arguments passed to the UNet model.

        Returns
        -------
        output : torch.Tensor, shape (n, 3, 32, 32)
            Output of the model.
        """
        k = self.scheduler.model_input(t)
        return cast("torch.Tensor", self.unet(x, k, **kwargs).sample)
