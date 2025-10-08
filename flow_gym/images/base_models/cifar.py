"""Pre-trained base model for CIFAR-10."""

from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from flow_gym import BaseModel, DiffusionScheduler, FGTensor
from flow_gym.registry import base_model_registry

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d import UNet2DModel


@base_model_registry.register("images/cifar")
class CIFARBaseModel(BaseModel[FGTensor]):
    """Pre-trained diffusion model on CIFAR-10 32x32.

    Uses the `google/ddpm-cifar10-32` model from the `diffusers` library.

    Examples
    --------
    ```python
    device = torch.device("cpu")
    base_model = CIFARBaseModel().to(device)
    reward = CompressionReward()
    env = EpsilonEnvironment(base_model, reward, discretization_steps=100)
    policy = copy.deepcopy(base_model)
    env.policy = policy
    ```
    """

    output_type = "epsilon"

    def __init__(self, device: Optional[torch.device]):
        super().__init__(device)

        pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
        self.unet: UNet2DModel = pipe.unet

        pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
        self._scheduler = DiffusionScheduler(pipe.scheduler.alphas_cumprod)

    @property
    def scheduler(self) -> DiffusionScheduler:
        """Scheduler used for sampling."""
        return self._scheduler

    def sample_p0(self, n: int) -> FGTensor:
        """Sample n datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : FGTensor, shape (n, 3, 32, 32)
            Samples from the base distribution :math:`p_0`.

        Notes
        -----
        The base distribution :math:`p_0` is a standard Gaussian distribution.
        """
        return FGTensor(torch.randn(n, 3, 32, 32, device=self.device))

    def forward(self, x: FGTensor, t: torch.Tensor, **kwargs: Any) -> FGTensor:
        r"""Forward pass of the model, outputting :math:`\epsilon(x_t, t)`.

        Parameters
        ----------
        x : FGTensor, shape (n, 3, 32, 32)
            Input data.
        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].
        **kwargs : dict
            Additional keyword arguments passed to the UNet model.

        Returns
        -------
        output : FGTensor, shape (n, 3, 32, 32)
            Output of the model.
        """
        k = self.scheduler.model_input(t)
        output = cast("torch.Tensor", self.unet(x, k, **kwargs).sample)
        return FGTensor(output)
