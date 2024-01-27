#!/usr/bin/env python
# coding: utf-8

# # tengu-py
#
# > Python SDK for the QDX Quantum Chemistry workflow management system

import asyncio
from datetime import datetime
from pathlib import Path
# from pdbtools import pdb_fetch, pdb_delhetatm

import rush

from .common import setup_workspace, check_status_and_report_failures


EXPERIMENT = "experiment-e2e-charge"
RCSB_ID = "3h7w"
TAGS = [EXPERIMENT, RCSB_ID]
WORKSPACE_DIR = Path.home() / "scratch" / "rush" / EXPERIMENT
SMALL_JOB_RESOURCES = rush.Resources(storage=100, storage_units="MB")
MACHINE_NAME = "NIX_SSH_2"

machine_data = {}
machine_data["target"] = MACHINE_NAME
if MACHINE_NAME == "GADI":
    # Gadi needs CPUs and walltime
    machine_data["resources"] = {
        "gpus": 4,
        "storage": 10,
        "storage_units": "GB",
        "cpus": 48,
        "walltime": 60,
    }
elif MACHINE_NAME == "NIX_SSH_2" or MACHINE_NAME == "NIX_SSH_3":
    # All the Nix machines care about
    machine_data["resources"] = {
        "gpus": 0,
        "storage": 10,
        "storage_units": "GB",
    }

# Set our inputs
PREPARED_PROTEIN_PDB_PATH = WORKSPACE_DIR / f"01_{RCSB_ID}_prepared_P.pdb"
PREPARED_LIGAND_PDB_PATH = WORKSPACE_DIR / f"01_{RCSB_ID}_prepared_L.pdb"


async def main(clean_workspace=False):
    # ## Build your client
    await setup_workspace(WORKSPACE_DIR, clean_workspace)
    client = await rush.build_provider_with_functions(
        workspace=WORKSPACE_DIR,
        batch_tags=TAGS,
    )

    # ## Input selection

    # # fetch datafiles
    # complex = list(pdb_fetch.fetch_structure(RCSB_ID))
    # protein = pdb_delhetatm.remove_hetatm(complex)
    # # write our files to the locations defined in the config block
    # with open(SYSTEM_PDB_PATH, "w") as f:
    #     for substructure in complex:
    #         f.write(str(substructure))
    # with open(PROTEIN_PDB_PATH, "w") as f:
    #     for substructure in protein:
    #         f.write(str(substructure))

    # # ## 1.1) Input Preparation

    # # ### 1.1.1) Prep the protein

    # (_prepared_protein_qdxf, prepared_protein_pdb) = await client.prepare_protein(
    #     PROTEIN_PDB_PATH,
    #     target="NIX_SSH_3",
    #     resources={"gpus": 1, "storage": 10, "storage_units": "GB", "cpus": 48, "walltime": 60},
    #     restore=False,
    # )
    # print(f"{datetime.now().time()} | Running protein prep!")
    # await check_status_and_report_failures(client)
    # await prepared_protein_pdb.download(filename=f"01_{RCSB_ID}_prepared_protein_allchains.pdb")
    # print(f"{datetime.now().time()} | Downloaded prepped protein! {prepared_protein_pdb}")

    ## 1.2) Run GROMACS (module: gmx_tengu / gmx_tengu_pdb)

    gmx_config = {
        "params_overrides": {
            "md": {
                "nsteps": 5000,
                "nstenergy": 5000,
                "nstlog": 5000,
                "nstxout-compressed": 5000,
            },
            "em": {"nsteps": 5000},
            "nvt": {"nsteps": 5000},
            "npt": {"nsteps": 5000},
        },
        "num_gpus": machine_data["resources"]["gpus"],
        "num_replicas": 1,
        "frame_sel": {"start_frame": 0, "end_frame": 10, "interval": 1},
        "ligand_charge": None,
        "save_wets": False,
    }
    (
        gros,
        tprs,
        tops,
        logs,
        dry_xtc,
        dry_frames,
        index,
        _wet_xtc,
    ) = await client.gmx(
        None,
        PREPARED_PROTEIN_PDB_PATH,
        PREPARED_LIGAND_PDB_PATH,
        gmx_config,
        target=machine_data["target"],
        resources=machine_data["resources"],
        restore=False,
    )
    print(f"{datetime.now().time()} | Running GROMACS simulation!")
    await check_status_and_report_failures(client)
    await gros.download(filename=f"02_{RCSB_ID}_gmx_gros.tar.gz")
    await tprs.download(filename=f"02_{RCSB_ID}_gmx_tprs.tar.gz")
    await tops.download(filename=f"02_{RCSB_ID}_gmx_tops.tar.gz")
    await logs.download(filename=f"02_{RCSB_ID}_gmx_logs.tar.gz")
    await dry_xtc.download(filename=f"02_{RCSB_ID}_gmx_dry_xtc.tar.gz")
    await dry_frames.download(filename=f"02_{RCSB_ID}_gmx_dry_frames.tar.gz")
    await index.download(filename=f"02_{RCSB_ID}_gmx_index.tar.gz")
    print(f"{datetime.now().time()} | Downloaded GROMACS output! {dry_frames}")


if __name__ == "__main__":
    asyncio.run(main())
