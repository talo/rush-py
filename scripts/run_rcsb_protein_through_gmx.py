#!/usr/bin/env python

import asyncio
from datetime import datetime
from pathlib import Path

from pdbtools import pdb_fetch, pdb_delhetatm

import rush

from scripts.common import (
    setup_workspace,
    check_status_and_report_failures,
    get_resources,
    extract_gmx_dry_frames,
)

# Define our project information
EXPERIMENT = "experiment-e2e"
RCSB_ID = "3h7w"
TAGS = [EXPERIMENT, RCSB_ID]
WORKSPACE_DIR = Path.home() / "scratch" / "rush" / EXPERIMENT
SMALL_JOB_RESOURCES = rush.Resources(storage=100, storage_units="MB")

# Set our inputs
SYSTEM_PDB_PATH = WORKSPACE_DIR / f"00_{RCSB_ID}_C.pdb"
PROTEIN_PDB_PATH = WORKSPACE_DIR / f"00_{RCSB_ID}_P.pdb"


async def main(clean_workspace=False):
    # ## Build your client

    await setup_workspace(WORKSPACE_DIR, clean_workspace)
    client = await rush.build_provider_with_functions(
        workspace=WORKSPACE_DIR,
        batch_tags=TAGS,
    )

    # ## Input selection

    # fetch datafiles
    complex = list(pdb_fetch.fetch_structure(RCSB_ID))
    protein = pdb_delhetatm.remove_hetatm(complex)
    # write our files to the locations defined in the config block
    with open(SYSTEM_PDB_PATH, "w") as f:
        for substructure in complex:
            f.write(str(substructure))
    with open(PROTEIN_PDB_PATH, "w") as f:
        for substructure in protein:
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
        None,
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

    # ## 1.3) Run HERMES (module: hermes_energy)

    hermes_target = "NIX_SSH_3"

    (converted_protein,) = await client.convert(
        "PDB",
        WORKSPACE_DIR / f"02_{RCSB_ID}_gmx_frame0.pdb",
        target=hermes_target,
        resources=SMALL_JOB_RESOURCES,
        restore=True,
    )
    print(f"{datetime.now().time()} | Running protein conversion!")

    (picked_protein,) = await client.pick_conformer(
        converted_protein,
        0,
        target=hermes_target,
        resources=SMALL_JOB_RESOURCES,
        restore=True,
    )
    print(f"{datetime.now().time()} | Picking protein!")

    (fragmented_protein,) = await client.fragment_aa(
        picked_protein,
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

    hermes_resources = get_resources(hermes_target, 4)
    (hermes_energy, _hermes_gradient) = await client.hermes_energy(
        fragmented_protein,
        {
            "basis": "cc-pVDZ",
            "aux_basis": "cc-pVDZ-RIFIT",
            "method": "RHF",
        },
        {
            "debug": {},
            "export": {},
            "frag": {
                "method": "MBE",
                "fragmentation_level": 1,
                "fragmented_energy_type": "TotalEnergy",
                "ngpus_per_node": hermes_resources["gpus"],
            },
            "guess": {},
            "scf": {
                "convergence_metric": "diis",
                "dynamic_screening_threshold_exp": 10,
                "niter": 40,
                "ndiis": 8,
                "scf_conv": 1e-6,
            },
        },
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
