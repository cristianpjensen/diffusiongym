"""Reward function for protein-ligand docking tasks using AutoDock-GPU."""

import os
from typing import Any, Optional

import torch
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem

from flowgym import reward_registry
from flowgym.molecules import BatchedMoleculeReward
from flowgym.utils import temporary_workdir


@reward_registry.register("molecules/protein_docking")
class ProteinDockingReward(BatchedMoleculeReward):
    """Reward function for protein-ligand docking tasks.

    Requires AutoDock-GPU to be installed, a prepared protein, and a pre-computed grid.
    Assuming you have a protein with a ligand included, you can split them using the following gist:
    https://gist.github.com/cristianpjensen/16024f1d576423ec510acdcabb2b74bd. Once they have been
    split, you need to prepare the protein and compute the grid maps:
    ```
    mk_prepare_receptor.py --read_pdb <protein>_protein.pdb -o <protein> -p -g \
        --box_enveloping <protein>_ligand.pdb --padding 5
    autogrid4 -p <protein>.gpf -l <protein>.glg
    ```
    Then specify the directory with all these files as `protein_dir`.

    Parameters
    ----------
    protein_dir : str
        Directory containing the protein structure files.

    atom_type_map : list[str], default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
        List of atom types corresponding to the node features in the input graph. The index of each
        atom type in the list corresponds to its integer representation in the node features.

    validate : bool, default=True
        Whether to validate molecules using RDKit's sanitization.

    check_fragmented : bool, default=True
        Whether to allow fragmented molecules (i.e., molecules with multiple disconnected
        components).

    relax : bool, default=False
        Whether to relax the geometry of molecules using MMFF optimization. If using `relax`, you
        probably should also use `validate`, since otherwise there is a high chance that your
        program will crash.
    """

    def __init__(
        self,
        protein_dir: str,
        atom_type_map: Optional[list[str]] = None,
        validate: bool = True,
        check_fragmented: bool = True,
        relax: bool = False,
    ):
        super().__init__(
            atom_type_map=atom_type_map,
            validate=validate,
            check_fragmented=check_fragmented,
            relax=relax,
        )
        self.protein_dir = os.path.abspath(protein_dir)

    def compute_rewards(
        self,
        mols: list[Chem.Mol],
        indices: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the docking score of the molecules to the specified protein.

        Parameters
        ----------
        mol : list[Chem.Mol]
            The molecules to compute the reward for.

        Returns
        -------
        rewards : torch.Tensor, shape (len(mols),)
            The computed rewards.
        """
        out = torch.zeros(len(mols))
        valids = torch.zeros(len(mols), dtype=torch.bool)
        file_list = []

        with temporary_workdir():
            os.mkdir("ligands")

            # 1. Prepare all ligands for docking and keep track of all files so we can batch the
            # autodock call using a file list
            mk_prep = MoleculePreparation()
            for i, mol in enumerate(mols):
                try:
                    molsetup_list = mk_prep(mol)
                    molsetup = molsetup_list[0]
                    pdbqt_string, success, _ = PDBQTWriterLegacy.write_string(molsetup)
                    if not success:
                        continue
                except Exception:
                    continue

                fname = f"ligand_{i:05d}.pdbqt"
                with open(os.path.join("ligands", fname), "w") as f:
                    f.write(pdbqt_string)

                file_list.append(fname)

            # 2. Write file list to disk
            with open("ligands/file_list.txt", "w") as f:
                f.write(str(os.path.join(self.protein_dir, "protein.maps.fld")) + "\n")
                for fname in file_list:
                    f.write(fname + "\n")

            # 3. Run autodock GPU in batch mode
            os.system("autodock_gpu_128wi -B ligands/file_list.txt > /dev/null 2>&1")

            # 4. Extract docking scores from output files
            for i in range(len(mols)):
                fname = f"ligands/ligand_{i:05d}.dlg"
                if not os.path.exists(fname):
                    valids[i] = False
                    continue

                with open(fname, "r") as f:
                    for line in f:
                        if line.strip().endswith("RANKING"):
                            parts = line.split()
                            binding_affinity = float(parts[3])
                            out[i] = -binding_affinity
                            valids[i] = True

        return out, valids
