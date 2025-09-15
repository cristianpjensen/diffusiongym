"""Compression-based reward implementations for flow_gym."""

import torch
import torchvision

from .base import NonDifferentiableReward


def _bits_per_pixel(imgs: torch.Tensor, quality_level: int) -> torch.Tensor:
    IMG_BATCH_NDIM = 4
    assert imgs.ndim == IMG_BATCH_NDIM, "imgs should be a batch of images with shape (B, C, H, W)"

    # encode each image to jpeg
    imgs_uint8 = (imgs.clamp(0, 1) * 255).to(torch.uint8)
    encoded_bytes = torchvision.io.encode_jpeg(list(imgs_uint8), quality=quality_level)

    # calculate reward as bits/pixel
    num_pixels = imgs.shape[2] * imgs.shape[3]
    encoded_bytes = torch.tensor(
        [len(b) for b in encoded_bytes], device=imgs.device, dtype=torch.float32
    )
    bits_per_pixel = encoded_bytes * 8.0 / num_pixels

    return bits_per_pixel


class IncompressionReward(NonDifferentiableReward):
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

    def __init__(self, quality_level: int):
        self.quality_level = quality_level

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
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
        return _bits_per_pixel(imgs, self.quality_level)


class CompressionReward(NonDifferentiableReward):
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

    def __init__(self, quality_level: int):
        self.quality_level = quality_level

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """Compute the compression reward for a batch of images.

        Parameters
        ----------
        imgs : tensor, shape (B, C, H, W), values in [0, 1]
            A batch of images.

        Returns
        -------
        rewards : torch.Tensor, shape (B,)
            Compression reward (negative bits per pixel) for each image.
        """
        return -_bits_per_pixel(imgs, self.quality_level)
