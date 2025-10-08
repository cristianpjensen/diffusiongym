"""Pre-trained base model for CIFAR-10."""

from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from flow_gym import BaseModel, DiffusionScheduler, FGTensor
from flow_gym.registry import base_model_registry

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d import UNet2DModel


@base_model_registry.register("images/sd2")
class SD2BaseModel(BaseModel[FGTensor]):
    """Pre-trained 512x512 text-to-image diffusion model.

    Uses the `stabilityai/stable-diffusion-2-base` model from the `diffusers` library.
    """

    output_type = "epsilon"

    def __init__(self, device: Optional[torch.device]):
        super().__init__(device)

        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to(
            device
        )
        self.pipe = pipe
        self.unet: UNet2DModel = pipe.unet

        self.channels: int = self.unet.config["in_channels"]
        self.latent_size: int = self.unet.config["sample_size"]

        pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
        self._scheduler = DiffusionScheduler(pipe.scheduler.alphas_cumprod)

    @property
    def scheduler(self) -> DiffusionScheduler:
        """Scheduler used for sampling."""
        return self._scheduler

    def sample_p0(self, n: int) -> FGTensor:
        """Sample n latent datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : FGTensor, shape (n, 4, 64, 64)
            Samples from the base distribution :math:`p_0`.

        Notes
        -----
        The base distribution :math:`p_0` is a standard Gaussian distribution.
        """
        return FGTensor(
            torch.randn(n, self.channels, self.latent_size, self.latent_size, device=self.device)
        )

    def preprocess(self, x: FGTensor, **kwargs: Any) -> tuple[FGTensor, dict[str, Any]]:
        """Encode the prompt (if provided instead of encoder_hidden_states).

        Parameters
        ----------
        x : FGTensor, shape (n, 4, 64, 64)
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
        n = x.shape[0]
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        prompt = kwargs.get("prompt")
        neg_prompt = kwargs.get("neg_prompt", "")
        cfg_scale = kwargs.get("cfg_scale", 0.0)

        if encoder_hidden_states is None and prompt is None:
            raise ValueError("Either encoder_hidden_states or a prompt needs to be provided.")

        if encoder_hidden_states is None:
            if not isinstance(prompt, list):
                prompt = [prompt] * n

            assert len(prompt) == n, (
                "The prompt must be a list of strings with length equal to the batch size."
            )

            if neg_prompt is not None:
                if not isinstance(neg_prompt, list):
                    neg_prompt = [neg_prompt] * n

                if len(neg_prompt) != n:
                    raise ValueError(
                        "The negative prompt must be a list of strings with length equal to the ",
                        "batch size.",
                    )

            prompt_embeds, neg_prompt_embeds = self.pipe.encode_prompt(
                prompt, self.device, 1, cfg_scale > 0, neg_prompt
            )
            encoder_hidden_states = prompt_embeds
            if neg_prompt_embeds is not None:
                encoder_hidden_states = torch.cat([prompt_embeds, neg_prompt_embeds], dim=0)

        return x, {
            "encoder_hidden_states": encoder_hidden_states,
            "cfg_scale": cfg_scale,
        }

    def postprocess(self, x: FGTensor) -> FGTensor:
        """Decode the images from the latent space.

        Parameters
        ----------
        x : FGTensor, shape (n, 4, 64, 64)
            Final sample in latent space.

        Returns
        -------
        decoded : FGTensor, shape (n, 3, 512, 512)
            Decoded images in pixel space.
        """
        # Do this one-by-one to save on a lot of VRAM
        x = x / self.pipe.vae.config.scaling_factor
        decoded = torch.cat([self.pipe.vae.decode(xi.unsqueeze(0)).sample for xi in x], dim=0)
        return FGTensor(decoded)

    def forward(self, x: FGTensor, t: torch.Tensor, **kwargs: Any) -> FGTensor:
        r"""Forward pass of the model, outputting :math:`\epsilon(x_t, t)`.

        Parameters
        ----------
        x : FGTensor, shape (n, 4, 64, 64)
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        **kwargs : dict
            Additional keyword arguments passed to the UNet model.

        Returns
        -------
        output : FGTensor, shape (n, 4, 64, 64)
            Output of the model.
        """
        x_tensor = x.as_subclass(torch.Tensor)
        k = self.scheduler.model_input(t)

        cfg_scale = kwargs.get("cfg_scale", 0.0)
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided in kwargs.")

        if cfg_scale <= 0:
            out = cast("torch.Tensor", self.unet(x_tensor, k, encoder_hidden_states).sample)
            return FGTensor(out)

        x_tensor = torch.cat([x_tensor, x_tensor], dim=0)
        k = torch.cat([k, k], dim=0)
        out = cast("torch.Tensor", self.unet(x_tensor, k, encoder_hidden_states).sample)

        # Classifier-free guidance
        cond, uncond = out.chunk(2)
        out = (cfg_scale + 1) * cond - cfg_scale * uncond

        return FGTensor(out)
