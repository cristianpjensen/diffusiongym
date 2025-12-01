"""Utility functions for molecular reward calculations."""

import os
from typing import Optional

from rdkit import Chem

from flowgym.utils import temporary_workdir


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


def relax_geometry(mol: Chem.Mol) -> Chem.Mol | None:
    """Relax the geometry of a molecule using GFN-FF optimization.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to relax.

    Returns
    -------
    relaxed_mol : Chem.Mol | None
        The molecule with relaxed geometry. If any runtime errors, returns None.
    """
    with temporary_workdir():
        # Write molecule to XYZ file
        xyz_file = "input.xyz"
        output_file = "xtbopt.xyz"

        # Convert RDKit mol to XYZ format
        mol_block = Chem.MolToXYZBlock(mol)
        with open(xyz_file, "w") as f:
            f.write(mol_block)

        # Run xtb with GFN-FF optimization
        os.system(f"xtb {xyz_file} --opt --gfnff  > /dev/null 2>&1")

        if not os.path.exists(output_file):
            return None

        # Load optimized structure back into RDKit
        optimized_mol = Chem.MolFromXYZFile(output_file)  # type: ignore
        if optimized_mol is None:
            return None

        # Copy the optimized coordinates to the original molecule
        # to preserve bond orders, aromaticity, etc.
        conf = optimized_mol.GetConformer()
        original_conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            original_conf.SetAtomPosition(i, pos)  # type: ignore

    return mol
