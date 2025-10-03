"""Compression-based reward implementations for flow_gym."""

import torch
import torchvision

from flow_gym.utils import FGTensor

from .base import Reward


def _bits_per_pixel(imgs: torch.Tensor, quality_level: int) -> torch.Tensor:
    IMG_BATCH_NDIM = 4
    assert imgs.ndim == IMG_BATCH_NDIM, "imgs should be a batch of images with shape (B, C, H, W)"

    batch_size = imgs.shape[0]
    num_pixels = imgs.shape[2] * imgs.shape[3]
    bpp = torch.zeros(batch_size, device=imgs.device)

    imgs_uint8 = (imgs.clamp(0, 1) * 255).to(torch.uint8).cpu()
    for i in range(batch_size):
        encoded_bytes = torchvision.io.encode_jpeg(imgs_uint8[i], quality=quality_level)
        bpp[i] = 8 * len(encoded_bytes) / num_pixels

    return bpp.to(imgs.device)


class IncompressionReward(Reward[FGTensor]):
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

    def __call__(self, imgs: FGTensor) -> torch.Tensor:
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
        return _bits_per_pixel(torch.Tensor(imgs), self.quality_level)


class CompressionReward(Reward[FGTensor]):
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

    def __call__(self, imgs: FGTensor) -> torch.Tensor:
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
        return -_bits_per_pixel(torch.Tensor(imgs), self.quality_level)
