"""Validity reward for molecules."""

from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem import QED

from flowgym.molecules.rewards.base import MoleculeReward
from flowgym.registry import reward_registry


@reward_registry.register("molecules/qed")
class QEDReward(MoleculeReward):
    """QED reward for molecules."""

    def __init__(self, atom_type_map: Optional[list[str]] = None) -> None:
        super().__init__(atom_type_map, True, True, False)

    def compute_reward(self, mol: Chem.Mol, **kwargs: Any) -> float | None:
        """Compute Quantitative Estimate of Drug-likeness (QED) score."""
        return QED.qed(mol)  # type: ignore
