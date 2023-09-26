#!/usr/bin/env python3

import os
import json
from pathlib import Path
from time import sleep
from typing import Literal

from tengu.api import Arg, Provider


def check_target(s: str | None) -> Literal["NIX", "GADI", "NIX_SSH", "SETONIX"] | None:
    match s:
        case None:
            return None
        case "NIX":
            return s
        case "GADI":
            return s
        case "NIX_SSH":
            return s
        case "SETONIX":
            return s
        case _:
            raise ValueError(f"Invalid target {s}")


API_URL = os.getenv("INTEGRATION_SERVER_URL") or "http://localhost:8080"
TOKEN = os.getenv("INTEGRATION_TOKEN") or "b52509e6-5e61-4ae8-b43a-35a0ade4d806"
TARGET: Literal["NIX", "GADI", "NIX_SSH", "SETONIX"] = check_target(os.getenv("INTEGRATION_TARGET")) or "GADI"
TARGET_GPUS = 4 if TARGET == "GADI" else 8 if TARGET == "SETONIX" else 1


# We run pytest directly for integration tests, so we don't get test data alongside and have to explicitly
# provide it
integration_pdb_env = os.getenv("INTEGRATION_PDB_PATH")
integration_gro_env = os.getenv("INTEGRATION_GMX_PATH")
integration_lig_env = os.getenv("INTEGRATION_LIG_PATH")
test_dir_path = Path(__file__).parent.resolve()
gmx_pdb_path = Path(integration_pdb_env) if integration_pdb_env else test_dir_path / "gmx.pdb"

gmx_gro_path = Path(integration_gro_env) if integration_gro_env else test_dir_path / "gmx.gro"

gmx_lig_path = Path(integration_lig_env) if integration_lig_env else test_dir_path / "gro_lig.sdf"


def init_client(test_url=API_URL, token=TOKEN) -> Provider:
    return Provider(token, test_url)


modules = None


def get_modules(client: Provider):
    module_list = next(client.latest_modules())
    if module_list:
        modules = {module["path"].split("#")[1]: module["path"] for module in module_list}
        return modules
    return None


def test_qp():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("qp_gen_inputs") is not None
    assert modules.get("hermes_energy") is not None
    assert modules.get("qp_collate") is not None
    res = client.qp_run(
        modules["qp_gen_inputs"],
        modules["hermes_energy"],
        modules["qp_collate"],
        client.upload_arg(gmx_pdb_path),
        client.upload_arg(gmx_gro_path),
        client.upload_arg(gmx_lig_path),
        Arg(value="sdf"),
        Arg(value="MOL"),
        amino_acids_of_interest=Arg(
            value=[
                ("ACE", 838),
            ]
        ),
        target=TARGET,
        tags=["integration_test"],
        resources={"gpus": TARGET_GPUS},
        autopoll=(100, 100),
    )
    if not isinstance(res, list):
        assert res["status"] == "COMPLETED"
    else:
        assert False


def test_gmx_tengu():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("gmx_tengu_pdb") is not None
    assert modules.get("gmx_mmpbsa_tengu") is not None
    assert modules.get("gmx_frame_select_pdb") is not None
    res = client.run(
        modules["gmx_tengu_pdb"],
        [
            client.upload_arg(test_dir_path / "simple_test" / "cdk2_fixed_renumbered_nohet.pdb"),
            client.upload_arg(test_dir_path / "simple_test" / "cdk2_lig_17.pdb"),
            Arg(
                value={
                    "param_overrides": {
                        "md": [
                            ["nsteps", "1000"],
                            ["dt", "0.002"],
                            ["nstenergy", "1"],
                            ["nstxout-compressed", "1"],
                            ["nstlog", "1"],
                        ],
                    },
                    "num_gpus": 0,
                    "num_replicas": 1,
                    "frame_sel": {"begin_time": 1, "end_time": 2, "delta_time": 1},
                    "ligand_charge": None,
                },
            ),
        ],
        target=TARGET,
        tags=["integration_test"],
        resources={"gpus": 0, "cpus": 48, "mem": 98, "walltime": 60},
    )
    print(res)
    sleep(100)

    # mmpbsa
    mmpbsa_res = client.run(
        modules["gmx_mmpbsa_tengu"],
        [Arg(id=res["outs"][0]["id"]), Arg(value=0), Arg(value=10), Arg(), Arg(value=48)],
        target=TARGET,
        tags=["integration_test"],
        resources={"gpus": 0, "cpus": 48, "mem": 98},
    )

    # select frames
    select_res = client.run(
        modules["gmx_frame_select_pdb"],
        [Arg(id=res["outs"][0]["id"]), Arg(value={"frame_selection": [120, 140, 160]})],
        target=TARGET,
        tags=["integration_test"],
        resources={"gpus": 0, "cpus": 48, "mem": 98},
    )

    res_poll = client.poll_module_instance(res["id"])
    assert res_poll["status"] == "COMPLETED"

    select_res_poll = client.poll_module_instance(select_res["id"])
    assert select_res_poll["status"] == "COMPLETED"

    mmpbsa_res_poll = client.poll_module_instance(mmpbsa_res["id"])
    assert mmpbsa_res_poll["status"] == "COMPLETED"


def test_convert():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("convert") is not None
    res = client.run(
        modules["convert"],
        [
            Arg(
                value="PDB",
            ),
            client.upload_arg(test_dir_path / "simple_test" / "cdk2_fixed_renumbered_nohet.pdb"),
        ],
        target="NIX",
        tags=["integration_test"],
        resources={"gpus": 0, "storage": 728930},
    )
    print(res)
    sleep(10)
    res_poll = client.poll_module_instance(res["id"], poll_rate=10)
    assert res_poll["status"] == "COMPLETED"


def test_hermes_density():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("hermes_energy") is not None

    default_model = {
        "method": "RIMP2",
        "basis": "cc-pVDZ",
        "aux_basis": "cc-pVDZ-RIFIT",
        "frag_enabled": True,
    }

    frag_keywords = {
        "method": "MBE",
        "fragmentation_level": 2,
        "ngpus_per_node": TARGET_GPUS,
        "monomer_cutoff": 15,
        "dimer_cutoff": 15,
        "dimer_mp2_cutoff": 15,
        "trimer_cutoff": 15,
        "trimer_mp2_cutoff": 15,
        "fragmented_energy_type": "InteractivityEnergy",
        "reference_fragment": 1,
    }

    scf_keywords = {
        "convergence_metric": "diis",
        "dynamic_screening_threshold_exp": 10,
        "ndiis": 8,
        "niter": 40,
        "scf_conv": 0.000001,
    }

    instance = client.run(
        modules["hermes_energy"],
        [
            Arg(value=json.load((test_dir_path / "conformer.json").open())),
            Arg(value=default_model),
            Arg(
                value={
                    "frag": frag_keywords,
                    "scf": scf_keywords,
                    "debug": {},
                    "guess": {},
                    "export": {
                        "coefficient_export": True,
                    },
                }
            ),
        ],
        TARGET,
        {"walltime": 40, "gpus": TARGET_GPUS, "storage": 1 * 1024 * 1024},
    )

    # keep polling to see if module_instance is successful. This may take a while, > 10 mins
    completed_instance = client.poll_module_instance(instance["id"], n_retries=10, poll_rate=100)

    # the result will be an object, so fetch from object store
    client.object(completed_instance["outs"][0]["id"])  # will return the json energy results


def test_fetch_result():
    client = init_client()

    json.dump(
        client.object("420bcfbf-face-42d8-9354-d5284791718a"),
        open("out_qdxf.json", "w"),
        indent=2,
        # sort_keys=True,
    )  # will return the json energy results
    assert False


def test_gnina():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("gnina_tengu_pdb") is not None
    res = client.run(
        modules["gnina_tengu_pdb"],
        [
            client.upload_arg(test_dir_path / "simple_test" / "cdk2_fixed_renumbered_nohet.pdb"),
            client.upload_arg(test_dir_path / "gro_lig.sdf"),
            client.upload_arg(test_dir_path / "gro_lig.sdf"),
            Arg(value={"exhaustiveness": 8, "num_modes": 10, "minimise": False}),
        ],
        target=TARGET,
        tags=["integration_test"],
        resources={"gpus": 1, "storage": 1 * 1024 * 1024},
    )
    print(res)
    sleep(10)
    res_poll = client.poll_module_instance(res["id"])
    assert res_poll["status"] == "COMPLETED"


def test_gmx_tengu_protein_only():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("gmx_tengu_pdb") is not None
    res = client.run(
        modules["gmx_tengu_pdb"],
        [
            client.upload_arg(test_dir_path / "simple_test" / "cdk2_fixed_renumbered_nohet.pdb"),
            Arg(value=None),
            Arg(
                value={
                    "param_overrides": {
                        "md": [
                            ("nsteps", "1000"),
                            ("dt", "0.002"),
                            ("nstenergy", "1"),
                            ("nstxout-compressed", "1"),
                            ("nstlog", "1"),
                        ],
                        "em": [],
                        "nvt": [],
                        "ions": [],
                        "npt": [],
                    },
                    "num_gpus": 0,
                    "num_replicas": 1,
                    "frame_sel": {"begin_time": 1, "end_time": 2, "delta_time": 1},
                    "ligand_charge": None,
                },
            ),
        ],
        target=TARGET,
        tags=["integration_test"],
        resources={"gpus": 0, "cpus": 48, "mem": 98, "walltime": 60},
    )
    print(res)
    sleep(100)

    res_poll = client.poll_module_instance(res["id"])
    assert res_poll["status"] == "COMPLETED"


def test_hermes_lattice():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("hermes_lattice") is not None

    default_model = {
        "method": "RIMP2",
        "basis": "cc-pVDZ",
        "aux_basis": "cc-pVDZ-RIFIT",
        "frag_enabled": True,
    }

    frag_keywords = {
        "dimer_cutoff": 25,
        "dimer_mp2_cutoff": 25,
        "fragmentation_level": 2,
        "method": "MBE",
        "monomer_cutoff": 30,
        "monomer_mp2_cutoff": 30,
        "ngpus_per_node": 1,
        "trimer_cutoff": 10,
        "trimer_mp2_cutoff": 10,
        "fragmented_energy_type": "TotalEnergy",  # should get overwritten by module
    }

    scf_keywords = {
        "convergence_metric": "diis",
        "dynamic_screening_threshold_exp": 10,
        "ndiis": 8,
        "niter": 40,
        "scf_conv": 0.000001,
    }

    instance = client.run(
        modules["hermes_lattice"],
        [
            Arg(value=json.load((test_dir_path / "conformer.json").open())),
            Arg(),
            Arg(value=default_model),
            Arg(value={"frag": frag_keywords, "scf": scf_keywords, "debug": {}, "guess": {}, "export": {}}),
        ],
        TARGET,
        {"walltime": 40, "gpus": TARGET_GPUS, "storage": 2 * 1024 * 1024},
    )

    # keep polling to see if module_instance is successful. This may take a while, > 10 mins
    completed_instance = client.poll_module_instance(instance["id"], n_retries=10, poll_rate=100)

    # the result will be an object, so fetch from object store
    client.object(completed_instance["outs"][0]["id"])  # will return the json energy results


def test_hermes_energy():
    client = init_client()
    modules = get_modules(client)
    assert modules
    assert modules.get("hermes_energy") is not None

    default_model = {
        "method": "RIMP2",
        "basis": "cc-pVDZ",
        "aux_basis": "cc-pVDZ-RIFIT",
        "frag_enabled": True,
    }

    frag_keywords = {
        "dimer_cutoff": 25,
        "dimer_mp2_cutoff": 25,
        "fragmentation_level": 2,
        "method": "MBE",
        "monomer_cutoff": 30,
        "monomer_mp2_cutoff": 30,
        "ngpus_per_node": 1,
        "reference_fragment": 1,
        "trimer_cutoff": 10,
        "trimer_mp2_cutoff": 10,
        "fragmented_energy_type": "InteractivityEnergy",
    }

    scf_keywords = {
        "convergence_metric": "diis",
        "dynamic_screening_threshold_exp": 10,
        "ndiis": 8,
        "niter": 40,
        "scf_conv": 0.000001,
    }

    instance = client.run(
        modules["hermes_energy"],
        [
            Arg(value=json.load((test_dir_path / "conformer.json").open())),
            Arg(value=default_model),
            Arg(value={"frag": frag_keywords, "scf": scf_keywords, "debug": {}, "guess": {}, "export": {}}),
        ],
        TARGET,
        {"walltime": 40, "gpus": TARGET_GPUS},
    )

    # keep polling to see if module_instance is successful. This may take a while, > 10 mins
    completed_instance = client.poll_module_instance(instance["id"], n_retries=10, poll_rate=100)

    # the result will be an object, so fetch from object store
    client.object(completed_instance["outs"][0]["id"])  # will return the json energy results


def test_retry():
    client = init_client()

    instance = next(client.module_instances())[0]
    # this means that the instance made some progress, so we can retry it
    client.retry(instance["id"])


def test_list_modules():
    client = init_client()

    modules = []
    for page in client.latest_modules():
        modules += [module for module in page]

    assert len(modules) > 0
