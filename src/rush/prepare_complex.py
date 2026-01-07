"""
Protein-ligand complex preparation module for the Rush Python client.

This module builds on the protein preparation workflow to prepare complexes by
extracting ligands from PDB inputs, adding hydrogens, and merging ligand data
with prepared protein TRC data for downstream computations.
"""

from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from rdkit import Chem

from rush import from_json, from_pdb, to_pdb
from rush.client import (
    RunOpts,
    RunSpec,
)
from rush.prepare_protein import prepare_protein as run_prepare_protein
from rush.prepare_protein import save_outputs as save_prepare_protein_outputs
from rush.trc.merge import merge_trcs


def _extract_ligand_with_hydrogens(pdb_path, ligand_resnames):
    """
    Load a PDB, extract a ligand by residue name, add hydrogens, and save.

    Args:
        pdb_path: Path to input PDB file
        ligand_resname: Residue name of the ligand (e.g., "LIG", "UNK", "ATP")
        output_path: Path for output PDB file
    """

    # Normalize to list
    if isinstance(ligand_resnames, str):
        ligand_resnames = [ligand_resnames]
    ligand_resnames = [name.strip() for name in ligand_resnames]

    # Load the PDB file
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not load PDB file: {pdb_path}")

    # Find atoms belonging to the ligand
    ligand_atom_indices = []
    for atom in mol.GetAtoms():
        res_info = atom.GetPDBResidueInfo()
        if res_info and res_info.GetResidueName().strip() in ligand_resnames:
            ligand_atom_indices.append(atom.GetIdx())

    if not ligand_atom_indices:
        raise ValueError(f"No residues '{ligand_resnames}' found in PDB")

    # Extract ligand as a new molecule
    ligand = Chem.RWMol(mol)
    atoms_to_remove = [
        i for i in range(mol.GetNumAtoms()) if i not in ligand_atom_indices
    ]
    for idx in sorted(atoms_to_remove, reverse=True):
        ligand.RemoveAtom(idx)

    ligand = ligand.GetMol()

    # Store residue info for each heavy atom before adding hydrogens
    # Map from atom idx -> residue info
    atom_res_info = {}
    for atom in ligand.GetAtoms():
        res_info = atom.GetPDBResidueInfo()
        if res_info:
            atom_res_info[atom.GetIdx()] = {
                "res_name": res_info.GetResidueName(),
                "chain": res_info.GetChainId(),
                "res_num": res_info.GetResidueNumber(),
                "insertion_code": res_info.GetInsertionCode(),
            }

    num_atoms_before = ligand.GetNumAtoms()

    # Add hydrogens with coordinates
    ligand_h = Chem.AddHs(ligand, addCoords=True)

    # Track hydrogen count per residue
    residue_h_count = defaultdict(int)

    # Assign residue info to new hydrogens based on their parent atom
    for atom in ligand_h.GetAtoms():
        if atom.GetIdx() >= num_atoms_before:  # This is a new hydrogen
            # Find the parent heavy atom
            neighbors = atom.GetNeighbors()
            if neighbors:
                parent_idx = neighbors[0].GetIdx()
                if parent_idx in atom_res_info:
                    info = atom_res_info[parent_idx]

                    # Create residue key for counting
                    res_key = (
                        info["chain"],
                        info["res_num"],
                        info["insertion_code"],
                        info["res_name"],
                    )
                    residue_h_count[res_key] += 1
                    h_num = residue_h_count[res_key]

                    # Create PDB residue info for the hydrogen
                    h_info = Chem.AtomPDBResidueInfo()
                    h_info.SetName(f" H{h_num}")
                    h_info.SetResidueName(info["res_name"])
                    h_info.SetChainId(info["chain"])
                    h_info.SetResidueNumber(info["res_num"])
                    h_info.SetInsertionCode(info["insertion_code"])
                    h_info.SetIsHeteroAtom(True)
                    h_info.SetOccupancy(1.0)
                    h_info.SetTempFactor(0.0)

                    atom.SetPDBResidueInfo(h_info)

    return Chem.MolToPDBBlock(ligand_h)


def prepare_complex(
    input_path: Path | str,
    ligand_names: list[str],
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run prepare-protein on a PDB or TRC file and return the separate T, R, and C files.
    """
    if isinstance(input_path, str):
        input_path = Path(input_path)

    if input_path.suffix == ".json":
        with NamedTemporaryFile(mode="w") as pdb_file:
            trc = from_json(input_path)
            if isinstance(trc, list):
                trc = trc[0]
            pdb_file.write(to_pdb(trc))
            pdb_l_str = _extract_ligand_with_hydrogens(pdb_file.name, ligand_names)
    else:
        pdb_l_str = _extract_ligand_with_hydrogens(input_path, ligand_names)

    trc_l = from_pdb(pdb_l_str)
    if isinstance(trc_l, list):
        trc_l = trc_l[0]

    res = run_prepare_protein(
        input_path,
        ph,
        naming_scheme,
        capping_style,
        truncation_threshold,
        run_spec,
        run_opts,
        collect,
    )
    trc_p_files = save_prepare_protein_outputs(res)
    trc_p = from_json(trc_p_files)
    if isinstance(trc_p, list):
        trc_p = trc_p[0]

    trc_c = merge_trcs(trc_p, trc_l)
    return trc_c


def save_outputs(res):
    return res
