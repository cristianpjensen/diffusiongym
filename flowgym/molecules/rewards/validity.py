"""Validity reward for molecules."""

from typing import Any, Optional

from rdkit import Chem

from flowgym.molecules.rewards.base import MoleculeReward
from flowgym.registry import reward_registry


@reward_registry.register("molecules/validity")
class ValidityReward(MoleculeReward):
    """Validity reward for molecules. It is 1 if chemically valid and no fragmentation, else 0."""

    def __init__(self, atom_type_map: Optional[list[str]] = None) -> None:
        super().__init__(atom_type_map, True, True, False)

    def compute_reward(self, mol: Chem.Mol, **kwargs: Any) -> float | None:
        """We only get here if the molecule is valid and non-fragmented."""
        return 1
