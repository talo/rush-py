#!/usr/bin/env python

import asyncio
import json
from datetime import datetime
from pathlib import Path

from pdbtools import pdb_fetch, pdb_delhetatm

import rush
import qdx_py

from .common import setup_workspace, check_status_and_report_failures

EXPERIMENT = "check-rcsb-to-hermes"
TAGS = [EXPERIMENT]
WORKSPACE_DIR = Path.home() / "scratch" / "rush" / EXPERIMENT
SMALL_JOB_RESOURCES = rush.Resources(storage=100, storage_units="MB")
MACHINE_NAME = "NIX_SSH_2"

# Set our inputs
PROTEIN_PDB_PATH = WORKSPACE_DIR / "test_P.pdb"


async def get_hermes_ready_conformer(client, rcsb_id):
    """
    Asynchronously start a protein preparation job
    """
    # ## Input selection

    # fetch datafiles
    complex = list(pdb_fetch.fetch_structure(rcsb_id))
    print(complex[0:10])
    protein = pdb_delhetatm.remove_hetatm(complex)
    # write our files to the locations defined in the config block
    with open(PROTEIN_PDB_PATH, "w") as f:
        for substructure in protein:
            f.write(str(substructure))

    # ## Prepare protein

    (prepared_protein_qdxf, _) = await client.prepare_protein(
        PROTEIN_PDB_PATH,
        tags=TAGS + [rcsb_id],
        target="NIX_SSH_2",
        resources=SMALL_JOB_RESOURCES,
        restore=True,
    )

    return prepared_protein_qdxf


async def main(rcsb_ids, clean_workspace=False):
    # ## Build your client

    await setup_workspace(WORKSPACE_DIR, clean_workspace)
    client = await rush.build_provider_with_functions(
        workspace=WORKSPACE_DIR,
        batch_tags=TAGS,
    )
    # print(await client.get_latest_module_paths())

    prep_outputs = [(rcsb_id, await get_hermes_ready_conformer(client, rcsb_id)) for rcsb_id in rcsb_ids]
    for rcsb_id in rcsb_ids:
        Path(client.workspace / f"objects/prepared_{rcsb_id}.qdxf.json").unlink(missing_ok=True)
    print(f"{datetime.now().time()} | Running protein prep!")
    await check_status_and_report_failures(client)
    await asyncio.gather(*[
        output[1].download(filename=f"prepared_{output[0]}.qdxf.json") for output in prep_outputs
    ])
    print(f"{datetime.now().time()} | Downloaded prepped proteins!")

    # ## Perceive bonds and charges for protein

    print(f"{datetime.now().time()} | Running protein bond and charge perception!")
    for rcsb_id in rcsb_ids:
        with open(client.workspace / f"objects/prepared_{rcsb_id}.qdxf.json", "r") as f:
            prepared_protein_qdxf = json.load(f)[0]
        charged_protein_qdxf = json.loads(qdx_py.formal_charge(json.dumps(prepared_protein_qdxf), "All"))
        with open(client.workspace / f"objects/charged_{rcsb_id}.qdxf.json", "w") as f:
            json.dump(charged_protein_qdxf, f, indent=2)
    print(f"{datetime.now().time()} | Saved charged protein!")

    # ## Validate protein

    print(f"{datetime.now().time()} | Running protein fragmentation!")
    (fragmented_protein,) = await client.fragment_aa(
        client.workspace / f"objects/charged_{rcsb_id}.qdxf.json",
        1,
        "All",
        tags=TAGS + [rcsb_id],
        target="NIX_SSH_2",
        resources=SMALL_JOB_RESOURCES,
        restore=True,
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
        tags=TAGS + [rcsb_id],
        target="GADI",
        resources=rush.Resources(gpus=4, storage=10, storage_units="GB", walltime=60),
        restore=True,
    )
    await check_status_and_report_failures(client)
    await hermes_energy.download(filename=f"02_hermes_energy_{rcsb_id}.json")


if __name__ == "__main__":
    asyncio.run(
        main([
            "3h7w",
            # "1alc",
            # "1b2a",
            # "1b9c",
            # "1cx8",
            # "1eyi",
            # "1fq4",
            # "1fz9",
            # "1g1x",
            # "1h1m",
            # "1hcb",
            # "1hjj",
            # "1j0c",
            # "1kjw",
            # "1kxy",
            # "1lth",
            # "1m5a",
            # "1nq1",
            # "1o2n",
            # "1o37",
            # "1obw",
            # "1oyl",
            # "1p11",
            # "1pfp",
            # "1pz2",
            # "1rh2",
            # "1tup",
            # "1w13",
            # "1wu6",
            # "1xxc",
            # "1ygs",
            # "1yxv",
            # "1znf",
            # "2c4v",
            # "2dbt",
            # "2ds0",
            # "2e6m",
            # "2eqo",
            # "2f8o",
            # "2hma",
            # "2i0r",
            # "2i87",
            # "2iq0",
            # "2l89",
            # "2lum",
            # "2mle",
            # "2mm0",
            # "2pbq",
            # "2pln",
            # "2pzs",
            # "2qgb",
            # "2r7k",
            # "2rdz",
            # "2w88",
            # "2wgl",
            # "2xbf",
            # "2y6l",
            # "2ziw",
            # "2zti",
            # "3awv",
            # "3c3l",
            # "3d1o",
            # "3dxn",
            # "3hra",
            # "3j2j",
            # "3j45",
            # "3jrs",
            # "3lo1",
            # "3mwk",
            # "3mxf",
            # "3n2z",
            # "3nld",
            # "3nnh",
            # "3obl",
            # "3ogs",
            # "3q11",
            # "3qdc",
            # "3r4c",
            # "3rge",
            # "3sc7",
            # "3t3z",
            # "3tyn",
            # "3u0b",
            # "3u2o",
            # "3ugk",
            # "4c0y",
            # "4c6q",
            # "4dmk",
            # "4ebv",
            # "4f5j",
            # "4fs7",
            # "4fyx",
            # "4gnk",
            # "4gud",
            # "4h2t",
            # "4hch",
            # "4hil",
            # "4hp3",
            # "4if6",
            # "4izo",
            # "4j9y",
            # "4loq",
            # "4mgq",
            # "4o6n",
            # "4olq",
            # "4pz6",
            # "4rx3",
            # "4z62",
            # "4zh6",
            # "5aa4",
            # "5e46",
            # "5eib",
            # "5et8",
            # "5exx",
            # "5fcb",
            # "5hp6",
            # "5hzm",
            # "5ii3",
            # "5jdn",
            # "5kr2",
            # "5lci",
            # "5ovk",
            # "5pp8",
            # "5px2",
            # "5qc9",
            # "5sjy",
            # "5ucv",
            # "5upw",
            # "5uqw",
            # "5wr5",
            # "5wru",
            # "5y0m",
            # "5y78",
            # "5zcd",
            # "6agr",
            # "6apz",
            # "6b5g",
            # "6b7g",
            # "6dde",
            # "6fib",
            # "6fie",
            # "6g37",
            # "6g89",
            # "6j82",
            # "6jcv",
            # "6jh0",
            # "6kbc",
            # "6kgj",
            # "6kps",
            # "6nci",
            # "6nk8",
            # "6q3a",
            # "6qet",
            # "6rme",
            # "6s9k",
            # "6tgx",
            # "6tiw",
            # "6u0w",
            # "6vok",
            # "6wnz",
            # "6wt2",
            # "6x5j",
            # "6yus",
            # "6yv7",
            # "6zwk",
            # "7apk",
            # "7cgz",
            # "7d8q",
            # "7dp6",
            # "7dtt",
            # "7enl",
            # "7fjl",
            # "7fnx",
            # "7g9n",
            # "7l67",
            # "7m99",
            # "7mx8",
            # "7n4u",
            # "7nfm",
            # "7o46",
            # "7ol4",
            # "7pes",
            # "7qa1",
            # "7re6",
            # "7rtn",
            # "7uze",
            # "7vh1",
            # "7vi5",
            # "7wr1",
            # "7xec",
            # "7ymy",
            # "8a1t",
            # "8aob",
            # "8ay1",
            # "8bh1",
            # "8dhh",
            # "8g6i",
            # "17gs",
            # "242l",
        ])
    )
