"""Dipole moment reward for molecules."""

import json
import os

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem

from flow_gym.molecules.rewards.base import MoleculeReward
from flow_gym.utils import temporary_workdir


class DipoleMomentReward(MoleculeReward):
    """Dipole moment reward for molecules. Requires xtb to be installed."""

    def compute_reward(self, mol: Chem.Mol) -> float:
        """Run GFN2-xTB to compute the dipole moment of the molecule."""
        return float(np.linalg.norm(run_xtb(mol).dipole)) * 2.5417


class XTBResult:
    """Class to parse the output of GFN2-xTB."""

    def __init__(self, filename: str):
        assert filename.endswith(".json"), f"Filename ({filename}) must end with .json"
        with open(filename, "r") as f:
            self.json_data = json.load(f)

    @property
    def energy(self) -> float:
        """Energy in Hartree."""
        return float(self.json_data["total energy"])

    @property
    def dipole(self) -> NDArray[np.float32]:
        """Dipole in Debye."""
        return np.array(self.json_data["dipole"])


def run_xtb(mol: Chem.Mol) -> XTBResult:
    """Run GFN2-xTB on a molecule."""
    # Check if xtb is installed
    if os.system("which xtb > /dev/null 2>&1") != 0:
        raise RuntimeError("xtb is not installed. Please install xtb to use this reward.")

    with temporary_workdir():
        Chem.MolToXYZFile(mol, "input.xyz")
        call = os.system("xtb input.xyz --json > input.out 2>&1")
        if call != 0:
            raise RuntimeError("xtb failed to run.")

        res = XTBResult("xtbout.json")

    return res
