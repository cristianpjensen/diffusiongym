"""Validity reward for molecules."""

from typing import List

from rdkit import Chem

from flow_gym.molecules.rewards.base import MoleculeReward


class ValidityReward(MoleculeReward):
    """Validity reward for molecules. It is 1 if chemically valid and no fragmentation, else 0."""

    def __init__(self, atom_type_map: List[str]) -> None:
        super().__init__(atom_type_map, True, True, False)

    def compute_reward(self, mol: Chem.Mol) -> float:
        """We only get here if the molecule is valid and non-fragmented."""
        return 1
