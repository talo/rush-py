"""
Protein-ligand complex preparation module for the Rush Python client.

This module builds on the protein preparation workflow to prepare complexes by
extracting ligands from PDB inputs, adding hydrogens, and merging ligand data
with prepared protein TRC data for downstream computations.

Usage::

    from rush import prepare

    result = prepare.protein_ligand("complex.pdb", ligand_names=["LIG"]).fetch()
    print(result.topology.symbols)

.. note::

    Unlike most modules, ``prepare.protein_ligand()`` runs a full pipeline
    internally (prepare protein, extract ligand, merge).  The returned
    :class:`~rush.run.RushRun` wraps the prepare-protein job; calling
    ``.fetch()`` or ``.save()`` blocks until that job completes, then
    performs the merge and returns the combined complex.
"""

from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from rdkit import Chem

from rush import TRC, TRCRef, from_json, from_pdb, merge_trcs, to_pdb
from rush.client import (
    RunOpts,
    RunSpec,
)
from rush.convert import _single_trc
from rush.run import RushRun

from ._protein import ResultRef
from ._protein import protein as _run_prepare_protein


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


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def protein_ligand(
    mol: TRC | Path | str,
    ligand_names: list[str],
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    opt: bool | None = None,
    debump: bool | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """
    Submit a complex preparation job for a PDB or TRC file.

    Internally runs prepare-protein, extracts ligands, merges, and uploads
    the combined TRC.  The returned :class:`~rush.run.RushRun` wraps the
    prepare-protein job.  Calling ``.fetch()`` or ``.save()`` blocks until
    the protein preparation completes, then performs the merge locally.

    Returns a :class:`~rush.run.RushRun` handle. Call ``.fetch()`` to get the
    parsed TRC, or ``.save()`` to write the output files to disk.
    """
    # TODO: Support all the input types that rush.prepare.protein() supports
    if isinstance(mol, str):
        mol = Path(mol)
        input_path = mol
    elif isinstance(mol, Path):
        input_path = mol

    if isinstance(mol, TRC) or (isinstance(mol, Path) and mol.suffix == ".json"):
        with NamedTemporaryFile(mode="w") as pdb_file:
            if isinstance(mol, TRC):
                trc = mol
            else:
                trc = from_json(mol)
            trc = _single_trc(trc, input_path)
            pdb_file.write(to_pdb(trc))
            pdb_l_str = _extract_ligand_with_hydrogens(pdb_file.name, ligand_names)
    else:
        pdb_l_str = _extract_ligand_with_hydrogens(input_path, ligand_names)

    trc_l = from_pdb(pdb_l_str)
    trc_l = _single_trc(trc_l, "ligand")

    # Submit prepare-protein
    pp_run = _run_prepare_protein(
        mol,
        ph,
        naming_scheme,
        capping_style,
        truncation_threshold,
        opt,
        debump,
        run_spec,
        run_opts,
    )

    # Return a wrapper RushRun that, when collected, waits for prepare-protein,
    # merges with ligand, uploads, and returns a ResultRef for the complex.
    return _ComplexRun(pp_run, trc_l)


class _ComplexRun(RushRun[ResultRef]):
    """RushRun subclass that performs the merge step on collect."""

    def __init__(self, pp_run: RushRun[ResultRef], trc_l: TRC) -> None:
        super().__init__(pp_run.id, ResultRef)
        self._pp_run = pp_run
        self._trc_l = trc_l

    @property
    def id(self):
        return self._pp_run.id

    def __repr__(self) -> str:
        return f"RushRun(id={self._pp_run.id!r})"

    def collect(self, max_wait_time: int = 3600) -> ResultRef:
        if self._collected is None:
            protein_trcs = self._pp_run.collect(max_wait_time=max_wait_time).fetch()
            uploaded = [
                TRCRef.upload(merge_trcs(trc_p, self._trc_l)) for trc_p in protein_trcs
            ]
            self._collected = ResultRef(models=uploaded)
        return self._collected
