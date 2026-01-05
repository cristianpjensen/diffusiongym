"""ImageReward from https://arxiv.org/abs/2304.05977."""

from typing import Any

import ImageReward as RewardModel  # type: ignore
import torch
from torchvision.transforms.functional import to_pil_image

from flowgym import FlowTensor, Reward
from flowgym.registry import reward_registry


@reward_registry.register("images/image_reward")
class ImageReward(Reward[FlowTensor]):
    """ImageReward that scores images based on a learned model of human preferences.

    Source: https://arxiv.org/abs/2304.05977
    """

    def __init__(self) -> None:
        self.reward_model = RewardModel.load("ImageReward-v1.0")

    def __call__(self, x: FlowTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the image reward for a batch of images.

        Parameters
        ----------
        x : tensor, shape (B, C, H, W), values in [0, 1]
            A batch of images.

        Returns
        -------
        rewards : torch.Tensor, shape (B,)
            Image rewards.
        """
        if "prompt" not in kwargs:
            raise ValueError("ImageReward requires a 'prompt' keyword argument.")

        if x.data.min() < 0 or x.data.max() > 1:
            raise ValueError(
                f"`x` must have values in [0, 1], got [{x.data.min()}, {x.data.max()}]"
            )

        prompt = kwargs["prompt"]

        with torch.autocast("cuda", enabled=False):
            pil_imgs = [to_pil_image(img.data.cpu().to(dtype=torch.float)) for img in x.data]

            rewards = []
            for prompt_txt, pil_img in zip(prompt, pil_imgs):
                rewards.append(self.reward_model.score(prompt_txt, pil_img))

        return torch.tensor(rewards), torch.ones(len(x), device=x.device)
