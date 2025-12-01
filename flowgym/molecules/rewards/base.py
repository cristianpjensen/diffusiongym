"""Base class for molecule rewards with basic functionalities."""

from abc import abstractmethod
from typing import Any, Optional

import dgl
import torch
from flowmol.analysis.molecule_builder import SampledMolecule
from rdkit import Chem, RDLogger

from flowgym import Reward
from flowgym.molecules.rewards.utils import non_fragmented, relax_geometry, validate_mol
from flowgym.molecules.types import FGGraph


class MoleculeReward(Reward[FGGraph]):
    """Base class for molecule rewards with basic functionalities.

    Parameters
    ----------
    atom_type_map : list[str], default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        List of atom types corresponding to the node features in the input graph. The index of each
        atom type in the list corresponds to its integer representation in the node features.

    validate : bool, default=False
        Whether to validate molecules using RDKit's sanitization.

    check_fragmented : bool, default=True
        Whether to allow fragmented molecules (i.e., molecules with multiple disconnected
        components).

    relax : bool, default=True
        Whether to relax the geometry of molecules using MMFF optimization. If using `relax`, you
        probably should also use `validate`, since otherwise there is a high chance that your
        program will crash.
    """

    def __init__(
        self,
        atom_type_map: Optional[list[str]] = None,
        validate: bool = False,
        check_fragmented: bool = True,
        relax: bool = True,
    ) -> None:
        RDLogger.DisableLog("rdApp.*")

        if atom_type_map is None:
            atom_type_map = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

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

        mol.UpdatePropertyCache(strict=False)

        if self.relax:
            return relax_geometry(mol)

        return mol

    @abstractmethod
    def compute_reward(self, mol: Chem.Mol, **kwargs: Any) -> float | None:
        """Compute the reward for the molecule.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to compute the reward for.

        Returns
        -------
        reward : float or None
            The computed reward.
        """
        ...

    def __call__(self, x: FGGraph, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
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
        valids = torch.zeros(x.graph.batch_size, device=x.graph.device, dtype=torch.bool)
        for g in dgl.unbatch(x.graph):
            i += 1
            mol = SampledMolecule(g.cpu(), self.atom_type_map).rdkit_mol
            if not isinstance(mol, Chem.Mol):
                valids[i] = False
                continue

            mol = self.preprocess(mol)
            if mol is None:
                valids[i] = False
                continue

            output = self.compute_reward(mol, **kwargs)
            if output is None:
                out[i] = 0
                valids[i] = False
            else:
                out[i] = output
                valids[i] = True

        return out, valids


class BatchedMoleculeReward(Reward[FGGraph]):
    """Base class for molecule rewards with basic functionalities.

    Parameters
    ----------
    atom_type_map : list[str], default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        List of atom types corresponding to the node features in the input graph. The index of each
        atom type in the list corresponds to its integer representation in the node features.

    validate : bool, default=False
        Whether to validate molecules using RDKit's sanitization.

    check_fragmented : bool, default=True
        Whether to allow fragmented molecules (i.e., molecules with multiple disconnected
        components).

    relax : bool, default=True
        Whether to relax the geometry of molecules using MMFF optimization. If using `relax`, you
        probably should also use `validate`, since otherwise there is a high chance that your
        program will crash.
    """

    invalid_reward: float = 0.0

    def __init__(
        self,
        atom_type_map: Optional[list[str]] = None,
        validate: bool = False,
        check_fragmented: bool = True,
        relax: bool = False,
    ) -> None:
        RDLogger.DisableLog("rdApp.*")

        if atom_type_map is None:
            atom_type_map = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

        self.atom_type_map = atom_type_map
        self.validate = validate
        self.check_fragmented = check_fragmented
        self.relax = relax

    def preprocess(self, mol: Chem.Mol) -> Chem.Mol | None:
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
            return relax_geometry(mol)

        return mol

    @abstractmethod
    def compute_rewards(
        self,
        mols: list[Chem.Mol],
        indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reward for the molecules.

        Parameters
        ----------
        mols : list[Chem.Mol]
            The molecules to compute the reward for.

        Returns
        -------
        reward : torch.Tensor, shape (len(mols),)
            The computed reward.
        """
        ...

    def __call__(self, x: FGGraph, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
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
        mols: list[Chem.Mol] = []
        indices = []
        for g in dgl.unbatch(x.graph):
            i += 1
            mol = SampledMolecule(g.cpu(), self.atom_type_map).rdkit_mol
            if not isinstance(mol, Chem.Mol):
                continue

            mol = self.preprocess(mol)
            if mol is None:
                continue

            mols.append(mol)
            indices.append(i)

        out = self.invalid_reward * torch.ones(x.graph.batch_size, device=x.graph.device)
        valids = torch.zeros(x.graph.batch_size, device=x.graph.device, dtype=torch.bool)

        rewards, v = self.compute_rewards(mols, indices, **kwargs)
        rewards = rewards.to(out.device)
        v = v.to(valids.device)
        out[indices] = rewards
        valids[indices] = v
        return out, valids
