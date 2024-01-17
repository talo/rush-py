#!/usr/bin/env python
# coding: utf-8

# # tengu-py
#
# > Python SDK for the QDX Quantum Chemistry workflow management system

import asyncio
import httpx
import json
import os
from datetime import datetime
from pathlib import Path

from pdbtools import (
    pdb_fetch,
    pdb_delhetatm,
    pdb_selchain,
    pdb_rplresname,
    pdb_keepcoord,
    pdb_selresname,
)

import tengu
import qdx_py


DESCRIPTION = "tengu-py demo notebook"
TAGS = ["checker-testing"]
WORK_DIR = Path.home() / "scratch" / "tengu" / "checker"
# Set our inputs
PROTEIN_PDB_PATH = WORK_DIR / "test_P.pdb"


async def check_status_and_report_failures(client):
    """
    This will show the status of all of your runs
    """
    await client.status()
    for instance_id, (status, name, count) in (await client.status()).items():
        print(f"{count}. {name}: {status}")
        if status.value == "FAILED":
            async for log_page in client.logs(instance_id, "stderr"):
                for log in log_page:
                    print(log)
        print()


async def split_complex(complex):
    complex_s = json.dumps(complex)
    protein = json.loads(qdx_py.drop_residues(complex_s, [0]))
    ligand = json.loads(qdx_py.drop_amino_acids(complex_s, list(range(len(complex["amino_acids"])))))
    return (protein, ligand)


async def get_hermes_ready_conformer(client, rcsb_id):
    """
    Asynchronously start a protein preparation job
    """
    # ## Input selection

    # # fetch datafiles
    # complex = list(pdb_fetch.fetch_structure(rcsb_id))
    # print(complex[0:10])
    # protein = pdb_delhetatm.remove_hetatm(complex)
    # # write our files to the locations defined in the config block
    # with open(PROTEIN_PDB_PATH, "w") as f:
    #     for substructure in protein:
    #         f.write(str(substructure))

    # ## Prepare protein

    prep_resources = tengu.Resources(gpus=1, storage=1, storage_units="GB", walltime=60)
    (prepared_protein_qdxf, _) = await client.prepare_protein(
        PROTEIN_PDB_PATH, target="NIX_SSH_2", resources=prep_resources, restore=False
    )

    return prepared_protein_qdxf


async def main(rcsb_ids):
    """
    Main function
    """
    # Ensure your workdir exists
    # if WORK_DIR.exists():
    #     client = tengu.Provider(workspace=WORK_DIR)
    #     await client.nuke()
    # os.makedirs(WORK_DIR)

    # ## Build your client

    client = await tengu.build_provider_with_functions(
        url=os.getenv("TENGU_URL"),
        access_token=os.getenv("TENGU_TOKEN"),
        workspace=WORK_DIR,
        batch_tags=TAGS,
    )
    # print(await client.get_latest_module_paths())

    print(f"{datetime.now().time()} | Running protein prep!")
    prep_outputs = [(rcsb_id, await get_hermes_ready_conformer(client, rcsb_id)) for rcsb_id in rcsb_ids]

    # ## Print tengu status

    status = await client.status(group_by="path")
    print(f"{'Module':<20} | {'Status':<20} | Count")
    print("-" * 50)
    for module, (status, path, count) in status.items():
        print(f"{path:<20} | {status:<20} | {count}")

    for rcsb_id in rcsb_ids:
        Path(client.workspace / f"objects/prepared_{rcsb_id}.qdxf.json").unlink(missing_ok=True)
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
    # TODO


if __name__ == "__main__":
    asyncio.run(main(["2zff", "1b39"]))
