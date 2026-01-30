"""QED reward for molecules."""

from typing import Any

import torch
from rdkit import RDLogger
from rdkit.Chem import QED

from flowgym.molecules.types import FlowGraph
from flowgym.registry import reward_registry
from flowgym.rewards import Reward

from .utils import graph_to_mols, is_not_fragmented, is_valid


@reward_registry.register("molecules/qed")
class QEDReward(Reward[FlowGraph]):
    """QED reward for molecules."""

    def __init__(self) -> None:
        RDLogger.DisableLog("rdApp.*")  # type: ignore

        # Int -> Atom type string for FlowMol models
        self.atom_type_map = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

    def __call__(self, sample: FlowGraph, latent: FlowGraph, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        mols = graph_to_mols(sample, self.atom_type_map)

        rewards = torch.zeros(len(sample), device=sample.device)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)
        for i, mol in enumerate(mols):
            if is_valid(mol) and is_not_fragmented(mol):
                rewards[i] = QED.qed(mol)
                valids[i] = True

        return rewards, valids
