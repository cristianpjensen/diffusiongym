"""Base class for molecule rewards with basic functionalities."""

from abc import abstractmethod
from typing import Optional

import dgl
import torch
from flowmol.analysis.molecule_builder import SampledMolecule
from rdkit import Chem, RDLogger

from flow_gym import Reward
from flow_gym.molecules.rewards.utils import non_fragmented, relax_geometry, validate_mol
from flow_gym.molecules.types import FGGraph


class MoleculeReward(Reward[FGGraph]):
    """Base class for molecule rewards with basic functionalities.

    Parameters
    ----------
    validate : bool, default=False
        Whether to validate molecules using RDKit's sanitization.

    fragmented : bool, default=True
        Whether to allow fragmented molecules (i.e., molecules with multiple disconnected
        components).

    relax : bool, default=True
        Whether to relax the geometry of molecules using MMFF optimization. If using `relax`, you
        probably should also use `validate`, since otherwise there is a high chance that your
        program will crash.
    """

    def __init__(
        self,
        atom_type_map: list[str],
        validate: bool = False,
        check_fragmented: bool = True,
        relax: bool = True,
    ) -> None:
        RDLogger.DisableLog("rdApp.*")

        self.atom_type_map = atom_type_map
        self.validate = validate
        self.check_fragmented = check_fragmented
        self.relax = relax

    def preprocess(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Preprocess the molecule by validation, fragmentation checking, and geometry relaxation.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to preprocess.

        Returns
        -------
        preprocessed_mol : Chem.Mol | None
            The preprocessed molecule if it passes all checks, else None.
        """
        if self.validate:
            mol = validate_mol(mol)  # type: ignore
            if mol is None:
                return None

        if self.check_fragmented:
            mol = non_fragmented(mol)  # type: ignore
            if mol is None:
                return None

        if self.relax:
            mol = relax_geometry(mol)

        return mol

    @abstractmethod
    def compute_reward(self, mol: Chem.Mol) -> float:
        """Compute the reward for the molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to compute the reward for.

        Returns
        -------
        reward : float
            The computed reward.
        """
        ...

    def __call__(self, x: FGGraph) -> torch.Tensor:
        """Compute the reward for the molecule.

        Parameters
        ----------
        mol : FGGraph
            The molecules to compute the reward for.

        Returns
        -------
        reward : torch.Tensor
            The computed reward.
        """
        i = -1
        out = torch.zeros(x.graph.batch_size, device=x.graph.device)
        for g in dgl.unbatch(x.graph):
            i += 1
            mol = SampledMolecule(g.cpu(), self.atom_type_map).rdkit_mol
            if not isinstance(mol, Chem.Mol):
                continue

            mol = self.preprocess(mol)
            if mol is None:
                continue

            out[i] = self.compute_reward(mol)

        return out
