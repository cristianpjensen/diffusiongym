"""Pre-trained base model for Diffusion Transformer."""

from typing import Any, Optional, cast

import torch
from diffusers.pipelines.dit.pipeline_dit import DiTPipeline

from flowgym import BaseModel, DiffusionScheduler, FGTensor
from flowgym.registry import base_model_registry


@base_model_registry.register("images/dit")
class DiTBaseModel(BaseModel[FGTensor]):
    """Pre-trained 256x256 ImageNet transformer diffusion model.

    Uses the `facebookresearch/DiT-XL-2-256` model from the `diffusers` library.
    """

    output_type = "epsilon"

    def __init__(self, device: Optional[torch.device]):
        super().__init__(device)

        pipe = DiTPipeline.from_pretrained(
            "facebook/DiT-XL-2-256",
        ).to(device)
        self.pipe = pipe
        self.transformer = pipe.transformer

        self.channels: int = self.transformer.config["in_channels"]
        self.dim: int = self.transformer.config["sample_size"]

        pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(device)
        self._scheduler = DiffusionScheduler(pipe.scheduler.alphas_cumprod)

    @property
    def scheduler(self) -> DiffusionScheduler:
        """Scheduler used for sampling."""
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FGTensor, dict[str, Any]]:
        """Sample n latent datapoints from the base distribution :math:`p_0`.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        samples : FGTensor, shape (n, 4, 64, 64)
            Samples from the base distribution :math:`p_0`.

        kwargs : dict
            Additional keyword arguments, a randomly selected class label is provided if
            "class_label" is not in the input.

        Notes
        -----
        The base distribution :math:`p_0` is a standard Gaussian distribution.
        """
        class_labels = kwargs.get("class_labels", None)
        cfg_scale = kwargs.get("cfg_scale", 0.0)

        # If no prompt is provided, sample them
        if class_labels is None:
            class_labels = torch.randint(0, 1000, (n,), device=self.device)

        if isinstance(class_labels, int):
            class_labels = torch.tensor([class_labels] * n, device=self.device)

        if isinstance(class_labels, list):
            class_labels = torch.tensor(class_labels, device=self.device)

        if len(class_labels) != n:
            raise ValueError(
                "The class_label must be a list of integers with length equal to the "
                f"batch size, got length {len(class_labels)}."
            )

        return (
            FGTensor(torch.randn(n, self.channels, self.dim, self.dim, device=self.device)),
            {"class_labels": class_labels, "cfg_scale": cfg_scale},
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
        class_labels = kwargs.get("class_labels", None)
        cfg_scale = kwargs.get("cfg_scale", 0.0)

        if class_labels is None:
            raise ValueError("class_labels must be provided in kwargs.")

        return x, {
            "cfg_scale": cfg_scale,
            "class_labels": class_labels,
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

        # Convert to [0, 1]
        decoded = (decoded + 1) / 2
        decoded = decoded.clamp(0, 1)

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
        class_labels = kwargs.get("class_labels")

        if class_labels is None:
            raise ValueError("class_labels must be provided in kwargs.")

        if isinstance(cfg_scale, torch.Tensor):
            use_cfg = torch.any(cfg_scale > 0).item()
        else:
            use_cfg = cfg_scale > 0

        if not use_cfg:
            out = cast("torch.Tensor", self.transformer(x_tensor, k, class_labels).sample[:, :4])
            return FGTensor(out)

        x_tensor = torch.cat([x_tensor, x_tensor], dim=0)
        k = torch.cat([k, k], dim=0)
        class_null = torch.zeros_like(class_labels)
        class_labels = torch.cat([class_labels, class_null], dim=0)
        out = cast("torch.Tensor", self.transformer(x_tensor, k, class_labels).sample)

        # Classifier-free guidance
        cond, uncond = out[:, :4].chunk(2)
        out = (cfg_scale + 1) * cond - cfg_scale * uncond

        return FGTensor(out)
