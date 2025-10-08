"""Utility functions for molecular reward calculations."""

from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem


def validate_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Validate molecules according to chemical validity.

    Source: https://arxiv.org/abs/2505.00518

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to validate.

    Returns
    -------
    validated_mol : Chem.Mol | None
        The sanitized molecule if valid, else None.
    """
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
        return None

    mol.UpdatePropertyCache(strict=True)
    return mol


def non_fragmented(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Return the original molecule if it is non-fragmented.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to check.

    Returns
    -------
    non_fragmented_mol : Chem.Mol | None
        The original molecule if it is fragmented, else None.
    """
    if len(Chem.GetMolFrags(mol)) == 1:
        return mol

    return None


def relax_geometry(mol: Chem.Mol) -> Chem.Mol:
    """Relax the geometry of a molecule using MMFF optimization.

    Parameters
    ----------
    mol : Chem.Mol

    Returns
    -------
    relaxed_mol : Chem.Mol
        The molecule with relaxed geometry.
    """
    AllChem.MMFFOptimizeMolecule(mol)
    return mol
