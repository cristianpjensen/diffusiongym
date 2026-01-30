"""Pre-trained base model for Stable Diffusion 2."""

import json
import random
from importlib.resources import open_text
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

from flowgym import BaseModel, DiffusionScheduler, FlowTensor
from flowgym.registry import base_model_registry

if TYPE_CHECKING:
    from diffusers.models.unets.unet_2d import UNet2DModel


class StableDiffusionBaseModel(BaseModel[FlowTensor]):
    """Stable Diffusion base model."""

    output_type = "epsilon"

    def __init__(
        self,
        model_name: str,
        cfg_scale: float = 0.0,
        prompts: Optional[list[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(device)

        pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)
        self.pipe = pipe
        self.unet: UNet2DModel = pipe.unet

        self.channels: int = self.unet.config["in_channels"]
        self.dim: int = self.unet.config["sample_size"]

        pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
        self._scheduler = DiffusionScheduler(pipe.scheduler.alphas_cumprod)

        self.cfg_scale = cfg_scale
        self.p_dropout = 0.1

        if prompts is None:
            with open_text("flowgym.images.base_models", "refl_data.json", encoding="utf-8") as f:
                refl_data = json.load(f)

            prompts = [item["text"] for item in refl_data]

        self.prompts = prompts

    @property
    def scheduler(self) -> DiffusionScheduler:
        """Scheduler used for sampling."""
        return self._scheduler

    @property
    def do_cfg(self) -> bool:
        return self.cfg_scale > 0.0

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        """Sample n latent datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        samples : FlowTensor, shape (n, 4, 64, 64)
            Samples from the base distribution :math:`p_0`.

        kwargs : dict
            Additional keyword arguments, a randomly selected prompt if not provided through the
            input.

        Notes
        -----
        The base distribution :math:`p_0` is a standard Gaussian distribution.
        """
        prompt = kwargs.get("prompt", None)

        # If no prompt is provided, sample them
        if prompt is None:
            prompt = random.choices(self.prompts, k=n)

        # If a single prompt is provided, replicate it
        if not isinstance(prompt, list):
            prompt = [prompt] * n

        if len(prompt) != n:
            raise ValueError(f"The prompt must be a list of strings with length equal to the batch size, got length {len(prompt)}.")

        return (
            FlowTensor(torch.randn(n, self.channels, self.dim, self.dim, device=self.device)),
            {"prompt": prompt},
        )

    def preprocess(self, x: FlowTensor, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        """Encode the prompt (if provided instead of encoder_hidden_states).

        Parameters
        ----------
        x : FlowTensor, shape (n, 4, 64, 64)
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
        n = len(x)
        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        prompt = kwargs.get("prompt")
        neg_prompt = kwargs.get("neg_prompt", "")

        if encoder_hidden_states is None and prompt is None:
            raise ValueError("Either encoder_hidden_states or a prompt needs to be provided.")

        if encoder_hidden_states is None:
            if not isinstance(prompt, list):
                prompt = [prompt] * n

            assert len(prompt) == n, "The prompt must be a list of strings with length equal to the batch size."

            if neg_prompt is not None:
                if not isinstance(neg_prompt, list):
                    neg_prompt = [neg_prompt] * n

                if len(neg_prompt) != n:
                    raise ValueError(
                        "The negative prompt must be a list of strings with length equal to the ",
                        "batch size.",
                    )

            prompt_embeds, neg_prompt_embeds = self.pipe.encode_prompt(prompt, self.device, 1, self.do_cfg, neg_prompt)
            encoder_hidden_states = prompt_embeds
            if neg_prompt_embeds is not None:
                encoder_hidden_states = torch.cat([prompt_embeds, neg_prompt_embeds], dim=0)
            else:
                encoder_hidden_states = prompt_embeds

        return x, {
            "encoder_hidden_states": encoder_hidden_states,
            "prompt": prompt,
            "neg_prompt": neg_prompt,
        }

    def postprocess(self, x: FlowTensor) -> FlowTensor:
        """Decode the images from the latent space.

        Parameters
        ----------
        x : FlowTensor, shape (n, 4, 64, 64)
            Final sample in latent space.

        Returns
        -------
        decoded : FlowTensor, shape (n, 3, 512, 512)
            Decoded images in pixel space.
        """
        # Do this one-by-one to save on a lot of VRAM
        x = x / self.pipe.vae.config.scaling_factor
        decoded = torch.cat([self.pipe.vae.decode(xi.unsqueeze(0)).sample for xi in x.data], dim=0)

        # Convert to [0, 1]
        decoded = (decoded + 1) / 2
        decoded = decoded.clamp(0, 1)

        return FlowTensor(decoded)

    def forward(self, x: FlowTensor, t: torch.Tensor, **kwargs: Any) -> FlowTensor:
        r"""Forward pass of the model, outputting :math:`\epsilon(x_t, t)`.

        Parameters
        ----------
        x : FlowTensor, shape (n, 4, 64, 64)
            Input data.

        t : torch.Tensor, shape (n,)
            Time steps, values in [0, 1].

        **kwargs : dict
            Additional keyword arguments passed to the UNet model.

        Returns
        -------
        output : FlowTensor, shape (n, 4, 64, 64)
            Output of the model.
        """
        x_tensor = x.data
        k = self.scheduler.model_input(t)

        encoder_hidden_states = kwargs.get("encoder_hidden_states")
        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided in kwargs.")

        if not self.do_cfg or self.training or encoder_hidden_states.shape[0] == x_tensor.shape[0]:
            out = cast("torch.Tensor", self.unet(x_tensor, k, encoder_hidden_states).sample)
            return FlowTensor(out)

        x_tensor = torch.cat([x_tensor, x_tensor], dim=0)
        k = torch.cat([k, k], dim=0)
        out = cast("torch.Tensor", self.unet(x_tensor, k, encoder_hidden_states).sample)

        # Classifier-free guidance
        cond, uncond = out.chunk(2)
        out = (self.cfg_scale + 1) * cond - self.cfg_scale * uncond

        return FlowTensor(out)

    def train_loss(
        self,
        x1: FlowTensor,
        xt: Optional[FlowTensor] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Add prompt dropout for training when CFG is enabled."""
        if self.do_cfg and self.p_dropout > 0:
            cond_embeds = kwargs["encoder_hidden_states"]
            uncond_embeds, _ = self.pipe.encode_prompt(
                prompt="",
                device=x1.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            uncond_embeds = uncond_embeds.expand(cond_embeds.shape[0], -1, -1)

            mask = torch.rand(cond_embeds.shape[0], device=cond_embeds.device) < self.p_dropout
            mask = mask[:, None, None]

            kwargs["encoder_hidden_states"] = torch.where(mask, uncond_embeds, cond_embeds)

        return super().train_loss(x1, xt, t, **kwargs)


@base_model_registry.register("images/sd2")
class SD2BaseModel(StableDiffusionBaseModel):
    """Pre-trained 512x512 Stable Diffusion 2 base."""

    def __init__(
        self,
        cfg_scale: float = 0.0,
        prompts: Optional[list[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            "PeggyWang/stable-diffusion-2-base",
            cfg_scale=cfg_scale,
            prompts=prompts,
            device=device,
        )


@base_model_registry.register("images/sd1.5")
class SD15BaseModel(StableDiffusionBaseModel):
    """Pre-trained 512x512 Stable Diffusion 1.5."""

    def __init__(
        self,
        cfg_scale: float = 0.0,
        prompts: Optional[list[str]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            "sd-legacy/stable-diffusion-v1-5",
            cfg_scale=cfg_scale,
            prompts=prompts,
            device=device,
        )
