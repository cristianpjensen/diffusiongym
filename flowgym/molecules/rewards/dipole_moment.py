"""Dipole moment reward for molecules."""

import json
import os
from json.decoder import JSONDecodeError
from typing import Any, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from rdkit import Chem

from flowgym.molecules.rewards.base import BatchedMoleculeReward
from flowgym.registry import reward_registry
from flowgym.utils import temporary_workdir


@reward_registry.register("molecules/dipole_moment")
class DipoleMomentReward(BatchedMoleculeReward):
    """Dipole moment reward for molecules. Requires xtb to be installed."""

    def compute_rewards(
        self,
        mols: list[Chem.Mol],
        indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GFN2-xTB to compute the dipole moments of the molecules in Debye."""
        results = parallel_xtb(mols, gfn2_opt=True)
        rewards = torch.tensor(
            [float(np.linalg.norm(res.dipole)) if res is not None else 0.0 for res in results],
            dtype=torch.float32,
        )
        rewards = rewards * 2.5417  # Convert from atomic units to Debye
        valids = torch.tensor([res is not None for res in results])

        return rewards, valids


@reward_registry.register("molecules/target_dipole_moment")
class TargetDipoleMomentReward(DipoleMomentReward):
    """MAE to a target dipole moment for molecules."""

    invalid_reward = -2.0

    def compute_rewards(
        self,
        mols: list[Chem.Mol],
        indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Absolute error to dipole moment target."""
        target: Optional[torch.Tensor] = kwargs.get("dipole_moment_target", None)
        if target is None:
            raise ValueError(
                "Target dipole moment must be provided "
                "as a keyword argument 'dipole_moment_target'."
            )

        target = target[indices]
        if target.shape[0] != len(mols):
            raise ValueError(
                f"Target dipole moment ({target.shape[0]}) must have the same length "
                f"as the number of molecules ({len(mols)})."
            )

        dipoles, valids = super().compute_rewards(mols, indices, **kwargs)
        rewards = -torch.abs(dipoles - target)

        return rewards, torch.tensor(valids, dtype=torch.bool)


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
        """Dipole in atomic units."""
        for key, value in self.json_data.items():
            if key.startswith("dipole"):
                return np.array(value)

        raise RuntimeError(
            "Dipole moment not found in xtb output. "
            f"Keys in output: {sorted(self.json_data.keys())}"
        )


def parallel_xtb(mols: list[Chem.Mol], gfn2_opt: bool = False) -> list[XTBResult | None]:
    """Run GFN2-xTB in parallel for a the molecules in the graph."""
    results = []
    with temporary_workdir():
        i = 0
        for mol in mols:
            i += 1
            Chem.MolToXYZFile(mol, f"{i}.xyz")

        ncpus = len(os.sched_getaffinity(0))

        # Compute properties using GFN2-xTB
        opt_str = "--opt" if gfn2_opt else ""
        os.system(
            f"parallel -j {ncpus} "
            f"'xtb {{}} {opt_str} --parallel 1 --namespace {{/.}} --json > {{/.}}.out 2>&1' "
            "::: *.xyz"
        )

        # Read results
        for i in range(1, len(mols) + 1):
            path = f"{i}.xtbout.json"

            try:
                res = XTBResult(path) if os.path.exists(path) else None
            except JSONDecodeError:
                res = None

            results.append(res)

    return results
