#!/usr/bin/env python

import asyncio
import json
from datetime import datetime
from pathlib import Path

from pdbtools import pdb_fetch, pdb_delhetatm, pdb_selchain, pdb_rplresname, pdb_keepcoord, pdb_selresname

import rush
import qdx_py

from scripts.common import (
    setup_workspace,
    check_status_and_report_failures,
    get_resources,
    extract_gmx_dry_frames,
)

# Define our project information
EXPERIMENT = "experiment-e2e-complex"
RCSB_ID = "1b39"
TAGS = [EXPERIMENT, RCSB_ID]
WORKSPACE_DIR = Path.home() / "scratch" / "rush" / EXPERIMENT
SMALL_JOB_RESOURCES = rush.Resources(storage=100, storage_units="MB")

# Set our inputs
SYSTEM_PDB_PATH = WORKSPACE_DIR / f"00_{RCSB_ID}_C.pdb"
PROTEIN_PDB_PATH = WORKSPACE_DIR / f"00_{RCSB_ID}_P.pdb"
LIGAND_PDB_PATH = WORKSPACE_DIR / f"00_{RCSB_ID}_L.pdb"
LIGAND_SMILES_STR = (
    "c1nc(c2c(n1)n(cn2)[C@H]3[C@@H]([C@@H]([C@H](O3)CO[P@@](=O)(O)O[P@](=O)(O)OP(=O)(O)O)O)O)N"
)


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

    # ## Fetch system (SMILES string is hardcoded for now)

    # fetch datafiles
    complex = list(pdb_fetch.fetch_structure(RCSB_ID))
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
        PROTEIN_PDB_PATH, target=prep_target, resources=SMALL_JOB_RESOURCES, restore=True
    )
    print(f"{datetime.now().time()} | Running protein prep!")
    await check_status_and_report_failures(client)
    Path(client.workspace / f"objects/01_{RCSB_ID}_prepared_protein.pdb").unlink(missing_ok=True)
    await prepared_protein_pdb.download(filename=f"01_{RCSB_ID}_prepared_protein.pdb")
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
        LIGAND_SMILES_STR,
        LIGAND_PDB_PATH,
        ligand_prep_config,
        target=prep_target,
        resources=SMALL_JOB_RESOURCES,
        restore=True,
    )
    print(f"{datetime.now().time()} | Running ligand prep!")
    Path(client.workspace / f"objects/01_{RCSB_ID}_prepared_ligand.pdb").unlink(missing_ok=True)
    Path(client.workspace / f"objects/01_{RCSB_ID}_prepared_ligand.sdf").unlink(missing_ok=True)
    await prepared_ligand_pdb.download(filename=f"01_{RCSB_ID}_prepared_ligand.pdb")
    await prepared_ligand_sdf.download(filename=f"01_{RCSB_ID}_prepared_ligand.sdf")
    print(f"{datetime.now().time()} | Downloaded prepped ligand! {prepared_ligand_sdf}")

    # ## 1.2) Run GROMACS (module: gmx_tengu)

    gmx_target = "NIX_SSH_2"
    gmx_resources = get_resources(gmx_target, 0)

    gmx_config = {
        "params_overrides": {
            "em": {"nsteps": 10000},
            "nvt": {"nsteps": 1000},
            "npt": {"nsteps": 1000},
            "md": {"nsteps": 1000},
            "ions": {},
        },
        "frame_sel": {"start_time_ps": 0, "end_time_ps": 10, "delta_time_ps": 1},
        "num_gpus": gmx_resources["gpus"],
        "num_replicas": 1,
        "save_wets": False,
        "ligand_charge": None,
    }
    (_gros, _tprs, _tops, _logs, _index, _dry_xtc, gmx_dry_frames, _wet_xtc) = await client.gmx(
        None,
        prepared_protein_pdb,
        prepared_ligand_pdb,
        gmx_config,
        target=gmx_target,
        resources=gmx_resources,
        restore=True,
    )
    print(f"{datetime.now().time()} | Running GROMACS simulation!")
    await check_status_and_report_failures(client)
    Path(client.workspace / f"objects/02_{RCSB_ID}_gmx_dry_frames.tar.gz").unlink(missing_ok=True)
    await gmx_dry_frames.download(filename=f"02_{RCSB_ID}_gmx_dry_frames.tar.gz")
    print(f"{datetime.now().time()} | Downloaded GROMACS output! {gmx_dry_frames}")

    extract_gmx_dry_frames(client, WORKSPACE_DIR / f"objects/02_{RCSB_ID}_gmx_dry_frames.tar.gz", RCSB_ID)

    # ## 1.3) Run quantum energy calculation (modules: qp_gen_inputs, hermes_energy, qp_collate)

    hermes_target = "NIX_SSH_3"

    (converted_system,) = await client.convert(
        "PDB",
        WORKSPACE_DIR / f"02_{RCSB_ID}_gmx_frame0.pdb",
        target=hermes_target,
        resources=SMALL_JOB_RESOURCES,
        restore=True,
    )
    print(f"{datetime.now().time()} | Running system conversion!")

    (picked_system,) = await client.pick_conformer(
        converted_system,
        0,
        target=hermes_target,
        resources=SMALL_JOB_RESOURCES,
        restore=True,
    )
    print(f"{datetime.now().time()} | Picking system!")

    # Split the complex; otherwise, fragment_aa will put each ligand atom into seperate fragments
    (picked_protein, picked_ligand) = split_complex(await picked_system.get())
    json.dump(picked_protein, open(WORKSPACE_DIR / f"03_{RCSB_ID}_prepared_protein.qdxf.json", "w"))
    json.dump(picked_ligand, open(WORKSPACE_DIR / f"03_{RCSB_ID}_prepared_ligand.qdxf.json", "w"))

    (fragmented_protein,) = await client.fragment_aa(
        WORKSPACE_DIR / f"03_{RCSB_ID}_prepared_protein.qdxf.json",
        1,
        "All",
        target=hermes_target,
        resources=SMALL_JOB_RESOURCES,
        restore=True,
    )
    print(f"{datetime.now().time()} | Running protein fragmentation!")
    await check_status_and_report_failures(client)
    Path(client.workspace / f"objects/03_{RCSB_ID}_fragmented_protein.qdxf.json").unlink(missing_ok=True)
    await fragmented_protein.download(filename=f"03_{RCSB_ID}_fragmented_protein.qdxf.json")
    print(f"{datetime.now().time()} | Downloaded fragmented protein! {gmx_dry_frames}")

    hermes_resources = get_resources(hermes_target, 1)
    (hermes_energy, _hermes_gradient) = await client.hermes_lattice(
        fragmented_protein,
        WORKSPACE_DIR / f"03_{RCSB_ID}_prepared_ligand.qdxf.json",
        None,
        None,
        target=hermes_target,
        resources=hermes_resources,
        restore=False,
    )
    print(f"{datetime.now().time()} | Running HERMES!")
    await check_status_and_report_failures(client)
    Path(client.workspace / f"objects/03_{RCSB_ID}_hermes_energy.json").unlink(missing_ok=True)
    await hermes_energy.download(filename=f"03_{RCSB_ID}_hermes_energy.json")
    print(f"{datetime.now().time()} | Downloaded HERMES output! {hermes_energy}")


if __name__ == "__main__":
    asyncio.run(main())
