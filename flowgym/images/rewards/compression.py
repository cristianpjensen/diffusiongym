"""Compression-based reward implementations."""

import io
from typing import Any

import torch
from torchvision.transforms.functional import to_pil_image

from flowgym import FlowTensor, Reward
from flowgym.registry import reward_registry


def _bits_per_pixel(imgs: torch.Tensor, quality_level: int) -> torch.Tensor:
    IMG_BATCH_NDIM = 4
    assert imgs.ndim == IMG_BATCH_NDIM, "imgs should be a batch of images with shape (B, C, H, W)"

    if imgs.min() < 0 or imgs.max() > 1:
        raise ValueError(f"`imgs` must have values in [0, 1], got [{imgs.min()}, {imgs.max()}]")

    batch_size = imgs.shape[0]
    pixels = imgs.shape[-1] * imgs.shape[-2]

    bpp = torch.zeros(batch_size, device=imgs.device)
    for i in range(batch_size):
        # Convert to PIL Image
        pil_image = to_pil_image(imgs[i].cpu().float())

        # Calculate compressed size (using JPEG) in bytes
        compressed_buffer = io.BytesIO()
        pil_image.save(compressed_buffer, format="JPEG", quality=quality_level, optimize=True)
        compressed_size = len(compressed_buffer.getvalue())

        # Convert to bits/pixel
        bpp[i] = compressed_size * 8 / pixels

    return bpp


@reward_registry.register("images/incompression")
class IncompressionReward(Reward[FlowTensor]):
    """Incompression reward for image models.

    Typically, when this reward is maximized, it encourages the model to produce images that have
    high detail and patterns.

    Parameters
    ----------
    quality_level : int, 1-100
        JPEG quality level. Lower values mean higher compression.

    Examples
    --------
    >>> reward = IncompressionReward(85)
    >>> imgs = torch.rand(8, 3, 256, 256)
    >>> result = reward(imgs)
    >>> result.shape
    torch.Size([8])
    """

    def __init__(self, quality_level: int = 85):
        self.quality_level = quality_level

    def __call__(self, x: FlowTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the incompression reward for a batch of images.

        Parameters
        ----------
        imgs : tensor, shape (B, C, H, W), values in [0, 1]
            A batch of images.

        Returns
        -------
        rewards : torch.Tensor, shape (B,)
            Incompression reward (bits per pixel) for each image.
        """
        return _bits_per_pixel(x.data, self.quality_level), torch.ones(len(x), device=x.device)


@reward_registry.register("images/compression")
class CompressionReward(Reward[FlowTensor]):
    """Compression reward for image models.

    Typically, when this reward is maximized, it encourages the model to produce images that look
    more vintage or like paintings.

    Parameters
    ----------
    quality_level : int, 1-100
        JPEG quality level. Lower values mean higher compression.

    Examples
    --------
    >>> reward = CompressionReward(85)
    >>> imgs = torch.rand(8, 3, 256, 256)
    >>> result = reward(imgs)
    >>> result.shape
    torch.Size([8])
    """

    def __init__(self, quality_level: int = 85):
        self.quality_level = quality_level

    def __call__(self, x: FlowTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the compression reward for a batch of images.

        Parameters
        ----------
        x : tensor, shape (B, C, H, W), values in [0, 1]
            A batch of images.

        Returns
        -------
        rewards : torch.Tensor, shape (B,)
            Compression reward (negative bits per pixel) for each image.
        """
        return -_bits_per_pixel(x.data, self.quality_level), torch.ones(len(x), device=x.device)
