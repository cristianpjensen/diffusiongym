"""LAION aesthetic score reward."""

import os
from typing import Any
from urllib.request import urlretrieve

import open_clip  # type: ignore
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from flowgym import FlowTensor, Reward
from flowgym.registry import reward_registry


@reward_registry.register("images/aesthetic")
class AestheticReward(Reward[FlowTensor]):
    """Aesthetic reward based on the aesthetic predictor from LAION."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._get_aesthetic_model("vit_l_14").to(self.device)
        self.clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            device=self.device,
        )
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=Image.BICUBIC, max_size=None, antialias=True),  # type: ignore
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __call__(self, sample: FlowTensor, latent: FlowTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the aesthetic score for a batch of images."""
        if sample.data.min() < 0 or sample.data.max() > 1:
            raise ValueError(f"`sample` must have values in [0, 1], got [{sample.data.min()}, {sample.data.max()}]")

        preprocessed_imgs = self.preprocess(sample.data)
        rewards = []
        for img_ in preprocessed_imgs:
            img = img_.to(self.device).unsqueeze(0)
            img_feats = self.clip.encode_image(img)  # type: ignore
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            rewards.append(self.model(img_feats))

        return torch.cat(rewards).squeeze(-1), torch.ones(len(sample), dtype=torch.bool)

    def _get_aesthetic_model(self, clip_model: str = "vit_l_14") -> nn.Module:
        """Source: https://github.com/LAION-AI/aesthetic-predictor/blob/main/asthetics_predictor.ipynb."""
        home = os.path.expanduser("~")
        cache_folder = home + "/.cache/emb_reader"
        path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
        if not os.path.exists(path_to_model):
            os.makedirs(cache_folder, exist_ok=True)
            url_model = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_" + clip_model + "_linear.pth?raw=true"
            urlretrieve(url_model, path_to_model)

        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError

        s = torch.load(path_to_model)
        m.load_state_dict(s)
        m.eval()
        return m
