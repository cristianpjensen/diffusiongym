"""Pre-trained base model for CIFAR-10."""

from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline

from flowgym import BaseModel, DiffusionScheduler, FlowTensor
from flowgym.registry import base_model_registry

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d import UNet2DModel


@base_model_registry.register("images/cifar")
class CIFARBaseModel(BaseModel[FlowTensor]):
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

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        """Sample n datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : FlowTensor, shape (n, 3, 32, 32)
            Samples from the base distribution :math:`p_0`.

        Notes
        -----
        The base distribution :math:`p_0` is a standard Gaussian distribution.
        """
        return FlowTensor(torch.randn(n, 3, 32, 32, device=self.device)), kwargs

    def postprocess(self, x: FlowTensor) -> FlowTensor:
        """Convert to [0, 1].

        Parameters
        ----------
        x : FlowTensor, shape (n, 3, 32, 32)
            Final sample in [-1, 1].

        Returns
        -------
        decoded : FlowTensor, shape (n, 3, 32, 32)
            Final sample in [0, 1].
        """
        return FlowTensor(((x.data + 1) / 2).clamp(0, 1))

    def forward(self, x: FlowTensor, t: torch.Tensor, **kwargs: Any) -> FlowTensor:
        r"""Forward pass of the model, outputting :math:`\epsilon(x_t, t)`.

        Parameters
        ----------
        x : FlowTensor, shape (n, 3, 32, 32)
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        **kwargs : dict
            Additional keyword arguments passed to the UNet model.

        Returns
        -------
        output : FlowTensor, shape (n, 3, 32, 32)
            Output of the model.
        """
        k = self.scheduler.model_input(t)
        output = cast("torch.Tensor", self.unet(x.data, k, **kwargs).sample)
        return FlowTensor(output)
