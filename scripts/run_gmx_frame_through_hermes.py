#!/usr/bin/env python
# coding: utf-8

import asyncio
from datetime import datetime
from pathlib import Path

import rush

from .common import setup_workspace, check_status_and_report_failures

EXPERIMENT = "debug-gmx-hermes-bridge"
SYSTEM = "quek_frame00"
TAGS = [EXPERIMENT, SYSTEM]
WORKSPACE_DIR = Path.home() / "scratch" / "rush" / EXPERIMENT
SMALL_JOB_RESOURCES = rush.Resources(storage=100, storage_units="MB")


async def main(clean_workspace=False):
    # ## Build your client
    await setup_workspace(WORKSPACE_DIR, clean_workspace)
    client = await rush.build_provider_with_functions(
        workspace=WORKSPACE_DIR,
        batch_tags=TAGS,
    )

    print(f"{datetime.now().time()} | Running protein conversion!")
    (converted_protein,) = await client.convert(
        "PDB",
        WORKSPACE_DIR / f"00_gmx_{SYSTEM}.pdb",
        resources=SMALL_JOB_RESOURCES,
    )

    print(f"{datetime.now().time()} | Picking protein!")
    (picked_protein,) = await client.pick_conformer(
        converted_protein,
        0,
        resources=SMALL_JOB_RESOURCES,
    )

    print(f"{datetime.now().time()} | Running protein fragmentation!")
    (fragmented_protein,) = await client.fragment_aa(
        picked_protein,
        1,
        "All",
        resources=SMALL_JOB_RESOURCES,
    )

    print(f"{datetime.now().time()} | Running HERMES!")
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
                "ngpus_per_node": 4,
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
        target="GADI",
        resources=rush.Resources(gpus=4, storage=10, storage_units="GB", walltime=60),
    )

    await check_status_and_report_failures(client)
    await hermes_energy.download(filename=f"02_hermes_energy_{SYSTEM}.json")


if __name__ == "__main__":
    asyncio.run(main())
