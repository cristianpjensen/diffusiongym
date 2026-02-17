"""Utility functions for molecular reward calculations."""

import dgl
from flowmol.analysis.molecule_builder import SampledMolecule
from rdkit import Chem
from rdkit.Chem import AllChem

from diffusiongym.molecules.types import DDGraph

ATOM_TYPE_MAP = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


def graph_to_mols(x: DDGraph) -> list[Chem.Mol]:
    """Convert a graph to molecules.

    Parameters
    ----------
    x : DDGraph
        The graph.

    Returns
    -------
    list[Chem.Mol]
        List of molecules in the graph.
    """
    mols = []
    for sample in dgl.unbatch(x.graph):
        mol = SampledMolecule(sample.cpu(), ATOM_TYPE_MAP).rdkit_mol
        mols.append(mol)

    return mols


def is_valid(mol: Chem.Mol) -> bool:
    """Validate molecules according to chemical validity.

    Source: https://arxiv.org/abs/2505.00518

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to validate.

    Returns
    -------
    bool
        True if the molecule is valid, False otherwise.
    """
    # sometimes it crashes randomly in C++, so guard it defensively just in case
    try:
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        Chem.AssignStereochemistryFrom3D(mol)

        for a in mol.GetAtoms():
            a.SetNoImplicit(True)  # type: ignore
            if a.HasProp("_MolFileHCount"):
                a.ClearProp("_MolFileHCount")

        flags = Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_ADJUSTHS  # type: ignore

        # Full sanitization, minus ADJUSTHS
        err = Chem.SanitizeMol(
            mol,
            sanitizeOps=flags,
            catchErrors=True,
        )

        # Non-zero bitmask means some step failed
        if err:
            return False

        mol.UpdatePropertyCache(strict=True)
        return True
    except Exception:
        return False


def is_not_fragmented(mol: Chem.Mol) -> bool:
    """Return True if the molecule is non-fragmented.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to check.

    Returns
    -------
    bool
        True if the molecule is non-fragmented, False otherwise.
    """
    if len(Chem.GetMolFrags(mol)) == 1:
        return True

    return False


def safe_mmff_relax(mol: Chem.Mol) -> Chem.Mol | None:
    """Safely relax a molecule using MMFF.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to relax.

    Returns
    -------
    Chem.Mol | None
        The relaxed molecule, or None if the relaxation failed.
    """
    try:
        AllChem.MMFFOptimizeMolecule(mol)  # type: ignore
        return mol
    except RuntimeError as e:
        if "Invariant Violation" in str(e):
            return None
        raise e
