"""GFN2-xTB reward for molecules."""

import json
import os
import re
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from flowgym.molecules.types import FlowGraph
from flowgym.registry import reward_registry
from flowgym.rewards import Reward
from flowgym.utils import temporary_workdir

from .utils import graph_to_mols, is_not_fragmented, is_valid


class XTBReward(Reward[FlowGraph]):
    """Reward based on properties computed with GFN2-xTB.

    Note: Requires xtb to be installed (version 6.5.1 recommended).
    """

    invalid_val = 0.0

    def __init__(self, attr_name: str, do_relax: bool = True):
        RDLogger.DisableLog("rdApp.*")  # type: ignore
        # Int -> Atom type string for FlowMol models
        self.atom_type_map = ["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]

        self.attr_name = attr_name

        if do_relax:
            self.relax = AllChem.MMFFOptimizeMolecule  # type: ignore
        else:
            self.relax = lambda x: x

    def __call__(self, sample: FlowGraph, latent: FlowGraph, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        mols = graph_to_mols(sample, self.atom_type_map)

        valid_mols = []
        valid_indices = []
        for i, mol in enumerate(mols):
            if is_valid(mol) and is_not_fragmented(mol):
                self.relax(mol)
                valid_mols.append(mol)
                valid_indices.append(i)

        if len(valid_mols) == 0:
            xtb_results = []
        else:
            xtb_results = parallel_xtb(valid_mols)

        rewards = self.invalid_val * torch.ones(len(sample), device=sample.device)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)
        for idx, res in zip(valid_indices, xtb_results):
            if res is None:
                continue

            reward_value = getattr(res, self.attr_name)
            rewards[idx] = reward_value
            valids[idx] = True

        return rewards, valids


@reward_registry.register("molecules/energy")
class EnergyReward(XTBReward):
    invalid_val = -500

    def __init__(self, do_relax: bool = True):
        super().__init__(attr_name="energy", do_relax=do_relax)


@reward_registry.register("molecules/homo")
class HOMOReward(XTBReward):
    invalid_val = -25

    def __init__(self, do_relax: bool = True):
        super().__init__(attr_name="homo", do_relax=do_relax)


@reward_registry.register("molecules/lumo")
class LUMOReward(XTBReward):
    invalid_val = -25

    def __init__(self, do_relax: bool = True):
        super().__init__(attr_name="lumo", do_relax=do_relax)


@reward_registry.register("molecules/homo_lumo_gap")
class HOMOLUMOGapReward(XTBReward):
    invalid_val = 0.0

    def __init__(self, do_relax: bool = True):
        super().__init__(attr_name="homo_lumo_gap", do_relax=do_relax)


@reward_registry.register("molecules/polarizability")
class PolarizabilityReward(XTBReward):
    invalid_val = 0.0

    def __init__(self, do_relax: bool = True):
        super().__init__(attr_name="polarizability", do_relax=do_relax)


@reward_registry.register("molecules/dipole_moment")
class DipoleMomentReward(XTBReward):
    invalid_val = 0.0

    def __init__(self, do_relax: bool = True):
        super().__init__(attr_name="dipole_moment", do_relax=do_relax)


@reward_registry.register("molecules/heat_capacity")
class HeatCapacityReward(XTBReward):
    invalid_val = 0.0

    def __init__(self, do_relax: bool = True):
        super().__init__(attr_name="heat_capacity", do_relax=do_relax)


class XTBResult:
    """Class to parse the output of GFN2-xTB."""

    def __init__(self, filename: Path):
        assert filename.suffix == ".json", f"Filename ({filename}) must end with .json"

        # Load JSON data
        with open(filename, "r") as f:
            self.data = json.load(f)

        # Load Log data (assumes .out file exists next to .json)
        # The parallel_xtb function saves JSON as *.xtbout.json and log as *.out
        log_filename = filename.parent / (filename.stem.split(".")[0] + ".out")

        if log_filename.exists():
            with open(log_filename, "r") as f:
                self.log_content = f.read()
        else:
            raise FileNotFoundError(f"Log file {log_filename} not found.")

    @property
    def energy(self) -> float:
        """Energy (Hartree)."""
        return float(self.data["total energy"])

    @property
    def homo(self) -> float:
        """Highest occupied molecular orbital (eV)."""
        occupation = np.asarray(self.data["fractional occupation"])
        energies = np.asarray(self.data["orbital energies/eV"])

        occupied_indices = np.where(occupation > 0)[0]
        if len(occupied_indices) == 0:
            raise ValueError("No occupied orbitals found.")

        highest_occupied_orbital = occupied_indices[-1]
        return float(energies[highest_occupied_orbital])

    @property
    def lumo(self) -> float:
        """Lowest unoccupied molecular orbital (eV)."""
        occupation = np.asarray(self.data["fractional occupation"])
        energies = np.asarray(self.data["orbital energies/eV"])

        unoccupied_indices = np.where(occupation == 0)[0]
        if len(unoccupied_indices) == 0:
            raise ValueError("No unoccupied orbitals found.")

        lowest_unoccupied_orbital = unoccupied_indices[0]
        return float(energies[lowest_unoccupied_orbital])

    @property
    def homo_lumo_gap(self) -> float:
        """HOMO-LUMO gap (eV)."""
        return self.lumo - self.homo

    @property
    def dipole_moment(self) -> float:
        """Dipole moment (Debye)."""
        return 2.5417 * float(np.linalg.norm(self.data["dipole"]))

    @property
    def polarizability(self) -> float:
        """Polarizability (Bohr^3)."""
        # Regex to find: Mol. α(0) /au      :         <val>
        match = re.search(r"Mol\.\s+α\(0\)\s+/au\s+:\s+([\d\.]+)", self.log_content)
        if not match:
            raise ValueError("Polarizability not found in log output.")

        return float(match.group(1))

    @property
    def heat_capacity(self) -> float:
        """Heat Capacity (cal/K/mol) at 298.15K."""
        # Look for the TOT line in the thermodynamic table.
        # Format: TOT     enthalpy    heat_capacity    entropy    entropy(J)
        # Example: TOT    5299.6013   30.4586          85.7819    358.9115
        # We search for TOT, skipping the enthalpy (first number), capturing the second number.
        # [+-]?\d+ matches integers, floats, scientific notation
        number_pattern = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
        pattern = rf"TOT\s+{number_pattern}\s+({number_pattern})"

        match = re.search(pattern, self.log_content)
        if not match:
            raise ValueError("Heat capacity (TOT row) not found in log output. Did you run with --ohess?")

        return float(match.group(1))


def parallel_xtb(mols: list[Chem.Mol]) -> list[XTBResult | None]:
    """Run GFN2-xTB in parallel for molecules in the graph."""
    results = []
    with temporary_workdir():
        i = 0
        for mol in mols:
            i += 1
            Chem.MolToXYZFile(mol, f"{i}.xyz")

        ncpus = len(os.sched_getaffinity(0))

        # Compute properties using GFN2-xTB
        # Added --hess to calculate Hessian (needed for Heat Capacity)
        os.system(f"parallel -j {ncpus} 'xtb {{}} --hess --parallel 1 --namespace {{/.}} --json > {{/.}}.out 2>&1' ::: *.xyz")

        # Read results
        for i in range(1, len(mols) + 1):
            path = Path(f"{i}.xtbout.json")

            try:
                res = XTBResult(path) if path.exists() else None
            except JSONDecodeError:
                res = None

            results.append(res)

    return results
