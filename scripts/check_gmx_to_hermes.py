#!/usr/bin/env python

import asyncio
import json
import tarfile
from datetime import datetime
from pathlib import Path

from pdbtools import pdb_fetch, pdb_delhetatm, pdb_selchain, pdb_rplresname, pdb_keepcoord, pdb_selresname

import rush
import qdx_py

from .common import setup_workspace, get_resources

# Define our project information
EXPERIMENT = "debug-gmx-hermes-bridge"
SYSTEM = "quek_frame00"
TAGS = [EXPERIMENT, SYSTEM]
WORKSPACE_DIR = Path.home() / "scratch" / "rush" / EXPERIMENT
SMALL_JOB_RESOURCES = rush.Resources(storage=100, storage_units="MB")

# Set our inputs
SYSTEM_PDB_PATH = WORKSPACE_DIR / "test_C.pdb"
PROTEIN_PDB_PATH = WORKSPACE_DIR / "test_P.pdb"
LIGAND_PDB_PATH = WORKSPACE_DIR / "test_L.pdb"
LIGAND_SMI_PATH = WORKSPACE_DIR / "test_L.smi"


def fix_gmx_output(client):
    # Extract the "dry" (i.e. non-solvated) pdb frames we asked for
    with tarfile.open(client.workspace / "objects" / "02_gmx_dry_frames.tar.gz", "r") as tf:
        selected_frame_pdbs = [tf.extractfile(member).read() for member in tf if "pdb" in member.name]
        for i, frame in enumerate(selected_frame_pdbs):
            with open(client.workspace / "objects" / f"02_gmx_output_frame_{i}.pdb", "w") as pf:
                print(frame.decode("utf-8"), file=pf)
    # Extract the ligand.gro file
    with tarfile.open(client.workspace / "objects" / "02_gmx_gros.tar.gz", "r") as tf:
        gro = [tf.extractfile(member).read() for member in tf if "temp" in member.name][0]
        with open(client.workspace / "objects" / "02_gmx_lig.gro", "w") as pf:
            print(gro.decode("utf-8"), file=pf)
    # Fix up the output from gromacs
    # prepared_gmx_protein_outs = await client.prepare_protein(
    #     client.workspace / "objects" / "02_gmx_output_frame_0.pdb", target="NIX_SSH_3", restore=False
    # )
    # return prepared_gmx_protein_outs


def split_complex(complex):
    complex_s = json.dumps(complex)
    protein = json.loads(qdx_py.drop_residues(complex_s, [0]))
    ligand = json.loads(qdx_py.drop_amino_acids(complex_s, list(range(len(complex["amino_acids"])))))
    return (protein, ligand)


async def main(clean_workspace=False):
    # ## Build your client
    await setup_workspace(WORKSPACE_DIR, clean_workspace)
    client = await rush.build_provider_with_functions(
        workspace=WORKSPACE_DIR,
        batch_tags=TAGS,
    )

    # ## Input selection

    # fetch datafiles
    complex = list(pdb_fetch.fetch_structure("1b39"))
    protein = pdb_delhetatm.remove_hetatm(pdb_selchain.select_chain(complex, "A"))
    # select the ATP residue
    ligand = pdb_selresname.filter_residue_by_name(complex, "ATP")
    # we require ligands to be labelled as UNL
    ligand = pdb_rplresname.rename_residues(ligand, "ATP", "UNL")
    # we don't want to repeat all of the remark / metadata that is already in the protein
    ligand = pdb_keepcoord.keep_coordinates(ligand)
    # write our files to the locations defined in the config block
    with open(SYSTEM_PDB_PATH, "w") as f:
        for substructure in complex:
            f.write(str(substructure))
    with open(PROTEIN_PDB_PATH, "w") as f:
        for substructure in protein:
            f.write(str(substructure))
    with open(LIGAND_PDB_PATH, "w") as f:
        for substructure in ligand:
            f.write(str(substructure))

    # ## 1.1) Input Preparation

    prep_target = "NIX_SSH_2"

    # ### 1.1.1) Prep the protein

    (_prepared_protein_qdxf, prepared_protein_pdb) = await client.prepare_protein(
        PROTEIN_PDB_PATH, target=prep_target, resources=SMALL_JOB_RESOURCES, restore=False
    )
    print(f"{datetime.now().time()} | Running protein prep!")
    await prepared_protein_pdb.download(filename="01_prepared_protein.pdb")
    print(f"{datetime.now().time()} | Downloaded prepped protein! {prepared_protein_pdb}")

    # ### 1.1.2) Prep the ligand

    ligand_prep_config = {
        "source": "",
        "output_folder": "./",
        "job_manager": "multiprocessing",
        "num_processors": -1,
        "max_variants_per_compound": 1,
        "thoroughness": 3,
        "separate_output_files": True,
        "min_ph": 6.4,
        "max_ph": 8.4,
        "pka_precision": 1.0,
        "skip_optimize_geometry": True,
        "skip_alternate_ring_conformations": True,
        "skip_adding_hydrogen": False,
        "skip_making_tautomers": True,
        "skip_enumerate_chiral_mol": True,
        "skip_enumerate_double_bonds": True,
        "let_tautomers_change_chirality": False,
        "use_durrant_lab_filters": True,
    }
    (prepared_ligand_pdb, prepared_ligand_sdf) = await client.prepare_ligand(
        Path(LIGAND_SMI_PATH).read_text(),
        LIGAND_PDB_PATH,
        ligand_prep_config,
        target=prep_target,
        resources=SMALL_JOB_RESOURCES,
        restore=False,
    )
    print(f"{datetime.now().time()} | Running ligand prep!")
    await prepared_ligand_pdb.download(filename="01_prepped_ligand.pdb")
    await prepared_ligand_sdf.download(filename="01_prepped_ligand.sdf")
    print(f"{datetime.now().time()} | Downloaded prepped ligand! {prepared_ligand_sdf}")

    # ## 1.2) Run GROMACS (module: gmx_tengu)

    gmx_target = "NIX_SSH_2"
    gmx_resources = get_resources(gmx_target, 0)
    gmx_config = {
        "params_overrides": {
            "md": {
                "nsteps": 5000,
                "nstenergy": 5000,
                "nstlog": 5000,
                "nstxout-compressed": 5000,
            },
            "em": {"nsteps": 10000},
            "nvt": {"nsteps": 5000},
            "npt": {"nsteps": 5000},
        },
        "num_gpus": gmx_resources["gpus"],
        "num_replicas": 1,
        "frame_sel": {"start_time_ps": 0, "end_time_ps": 10, "delta_time_ps": 1},
        "ligand_charge": None,
    }
    (gros, _tprs, _tops, _logs, _index, _dry_xtc, dry_frames, _wet_xtc) = await client.gmx(
        None,
        prepared_protein_pdb,
        prepared_ligand_pdb,
        gmx_config,
        target=gmx_target,
        resources=gmx_resources,
        restore=True,
    )
    print(f"{datetime.now().time()} | Running GROMACS simulation!")
    await dry_frames.download(filename="02_gmx_dry_frames.tar.gz")
    await gros.download(filename="02_gmx_gros.tar.gz")
    print(f"{datetime.now().time()} | Downloaded GROMACS output! {dry_frames} {gros}")

    """
    # TODO: Remove need for this
    (prepared_gmx_protein_qdxf, prepared_gmx_protein) = fix_gmx_output(client)
    print(f"{datetime.now().time()} | Finished re-prepping protein after GROMACS run! {prepared_gmx_protein}")

    # ## 1.3) Run quantum energy calculation (modules: qp_gen_inputs, hermes_energy, qp_collate)

    # Split the complex; otherwise, fragment_aa will put each ligand atom into seperate fragments
    (protein, ligand) = split_complex((await prepared_gmx_protein_qdxf.get())[0])
    json.dump(protein, open(WORKSPACE_DIR / "objects" / "prepared_gmx_protein.qdxf.json", "w"))
    json.dump(ligand, open(WORKSPACE_DIR / "objects" / "prepared_gmx_ligand.qdxf.json", "w"))
    (fragmented,) = await client.fragment_aa(
        WORKSPACE_DIR / "objects" / "prepared_gmx_protein.qdxf.json", 1, "All", resources={"storage": 1000000}
    )
    hermes_result = await client.hermes_lattice(
        fragmented,
        WORKSPACE_DIR / "objects" / "prepared_gmx_ligand.qdxf.json",
        None,
        None,
        target="NIX_SSH_3",
        resources={"storage": 10, "storage_units": "MB"},
    )
    await hermes_result[0].get()
    await hermes_result.get()
    print(f"{datetime.now().time()} | Got qp interaction energy! {hermes_result}")
    """


if __name__ == "__main__":
    asyncio.run(main())
