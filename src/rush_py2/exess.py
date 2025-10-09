#!/usr/bin/env python3
import json
import tarfile
import time
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Literal

import cyclopts
import h5py
import requests
import zstandard as zstd
from gql import Client, FileVar, gql
from gql.transport.exceptions import TransportQueryError
from gql.transport.requests import RequestsHTTPTransport

GRAPHQL_ENDPOINT = (
    "https://tengu-server-staging-seaography-720805281970.asia-southeast1.run.app"
)
API_KEY = "1f6904ca-a882-4ca0-b3f8-e0ca610317bf"
PROJECT_ID = "1dfc23a2-2ecf-44a0-b111-c9b8a573c98e"
MODULE_LOCK = {
    "exess_rex": "github:talo/tengu-exess/9ccfa0a22d6395a34e03121b68fd7c4661722650#exess_rex",
}
# GRAPHQL_ENDPOINT = (
#     "https://tengu-server-prod-seaography-720805281970.asia-southeast1.run.app"
# )
# API_KEY = "6b5428a1-ca95-4b0d-8c6a-d6a197b36f13"
# PROJECT_ID = "ba9a5fc0-24dc-4a51-a755-95a6b432c39f"
# MODULE_LOCK = {
#     "exess_rex": "github:talo/tengu-exess/66b121b25545069355d813c0305508e5b63251fb#exess_rex",
# }
client = Client(
    transport=RequestsHTTPTransport(
        url=GRAPHQL_ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}"},
    ),
)

runspec = Template("""RunSpec {
        resources = Resources {
          walltime = None,
          storage = Some 10,
          storage_units = Some MemUnits::MB,
          storage_mounts = None,
          cpus = None,
          mem = None,
          mem_units = None,
          gpus = Some 1,
          gpu_mem = None,
          gpu_mem_units = None,
          nodes = None,
          internet_access = None,
        },
        target = $target
      }""")

INITIAL_POLL_INTERVAL = 0.5
MAX_POLL_INTERVAL = 30
BACKOFF_FACTOR = 1.5
MAX_WAIT_TIME = 3600

type MethodT = Literal[
    "RestrictedHF",
    "UnrestrictedHF",
    "RestrictedRIMP2",
    "UnrestrictedRIMP2",
]

type BasisT = Literal[
    "3-21G",
    "4-31G",
    "5-21G",
    "6-21G",
    "6-31G",
    "6-311G",
    "6-31G(2df,p)",
    "6-31G(3df,3pd)",
    "6-31G*",
    "6-31G**",
    "6-31+G",
    "6-31+G*",
    "6-31+G**",
    "6-31++G",
    "6-31++G*",
    "6-31++G**",
    "PCSeg-0",
    "PCSeg-1",
    "STO-2G",
    "STO-3G",
    "STO-4G",
    "STO-5G",
    "STO-6G",
    "aug-cc-pVDZ",
    "aug-cc-pVTZ",
    "cc-pVDZ",
    "cc-pVTZ",
]

type AuxBasisT = Literal[
    "6-31G**-RIFIT",
    "6-311G**-RIFIT",
    "aug-cc-pVDZ-RIFIT",
    "aug-cc-pVTZ-RIFIT",
    "cc-pVDZ-RIFIT",
    "cc-pVTZ-RIFIT",
]

type FragmentLevelT = Literal[
    "Monomer",
    "Dimer",
    "Trimer",
    "Tetramer",
]


def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d]
    else:
        return d


def optional_str(v: str | int | float | list[int] | bool | None):
    return f"Some {v}" if v else "None"


def upload_object(project_id, filepath):
    mutation = gql(
        """
        mutation UploadObject($file: Upload!, $typeinfo: Json!, $format: ObjectFormat!, $project_id: String) {
            upload_object(file: $file, typeinfo: $typeinfo, format: $format, project_id: $project_id) {
                id
                object {
                    path
                    size
                    format
                }
                base_url
                url
            }
        }
     """
    )
    with filepath.open(mode="rb") as f:
        if filepath.suffix == ".json":
            mutation.variable_values = {
                "file": FileVar(f),
                "format": "Json",
                "typeinfo": {
                    "k": "record",
                    "t": {},
                },
                "project_id": project_id,
            }
        else:
            mutation.variable_values = {
                "file": FileVar(f),
                "format": "Bin",
                "typeinfo": {
                    "k": "record",
                    "t": {
                        "size": "u32",
                        "path": {
                            "k": "@",
                            "t": "$Bytes",
                        },
                    },
                    "n": "Object",
                },
                "project_id": project_id,
            }
        result = client.execute(mutation, upload_files=True)

    obj = result["upload_object"]["object"]
    print(f"Object uploaded: {obj}")
    return obj


def download_object(path):
    query = gql(
        """
        query GetObject($path: String!) {
            object_path(path: $path) {
                url
                object {
                    format
                    size
                }
            }
        }
    """
    )
    query.variable_values = {"path": path}

    result = client.execute(query)
    obj_descriptor = result["object_path"]

    # Json
    if obj_descriptor.get("contents") is not None:
        return obj_descriptor["contents"]
    # Bin
    elif obj_descriptor.get("url"):
        response = requests.get(obj_descriptor["url"])
        response.raise_for_status()
        return response.content

    raise Exception(f"Object at path {path} has neither contents nor URL")


def get_module_instance(run_id):
    query = gql(
        """
query GetModuleInstances($run_id: TextFilterInput!) {
    module_instances(filters: {run_id: $run_id}) {
        nodes {
            created_at
            admitted_at
            dispatched_at
            queued_at
            run_at
            completed_at
            deleted_at
            progress {
                n
                n_expected
                n_max
                done
            }
            status
            failure_reason
            failure_context {
                stdout
                stderr
                syserr
            }
        }
    }
}
"""
    )
    query.variable_values = {"run_id": {"eq": run_id}}

    result = client.execute(query)
    return result["module_instances"]["nodes"]


def fetch_results(run_id):
    query = gql(
        """
query GetResults($id: String!) {
    run(id: $id) {
        id
        status
        result
        stdout
    }
}
"""
    )
    query.variable_values = {"id": run_id}

    result = client.execute(query)
    return result["run"]


def submit_rex(project_id: str, rex: str):
    mutation = gql(
        """
        mutation EvalRex($input: CreateRun!) {
            eval(input: $input) {
                id
                status
            }
        }
    """
    )
    mutation.variable_values = {
        "input": {
            "rex": rex,
            "project_id": project_id,
            "module_lock": MODULE_LOCK,
            "draft": False,
        },
    }

    result = client.execute(mutation)
    run_id = result["eval"]["id"]
    print(f"Run submitted with ID: {run_id}")

    query = gql(
        """
        query GetStatus($id: String!) {
            run(id: $id) {
                status
            }
        }
    """
    )
    query.variable_values = {"id": run_id}

    start_time = time.time()
    poll_interval = INITIAL_POLL_INTERVAL
    while time.time() - start_time < MAX_WAIT_TIME:
        time.sleep(poll_interval)

        result = client.execute(query)
        status = result["run"]["status"]
        print(f"Status: {status}")
        module_instances = get_module_instance(run_id)
        if module_instances:
            print(f"Module status: {module_instances[0]}")

        if status == "done" or status == "error" or status == "cancelled":
            return fetch_results(run_id)

        poll_interval = min(poll_interval * BACKOFF_FACTOR, MAX_POLL_INTERVAL)

    raise Exception(f"Timeout: Run did not complete within {MAX_WAIT_TIME} seconds")


def energy(
    topology_path: Path,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT = "cc-pVDZ-RIFIT",
    fragment_level: FragmentLevelT = "Monomer",
    dimer_cutoff: float = 100.0,
    trimer_cutoff: float = 25.0,
    tetramer_cutoff: float = 10.0,
    target: Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"] | None = None,
):
    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_rex_s
      ($runspec)
      (exess_rex::ExessParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = exess_rex::Model {
          method = exess_rex::Method::$method,
          basis = "$basis",
          aux_basis = Some "$aux_basis",
          standard_orientation = Some exess_rex::StandardOrientation::FullSystem,
          force_cartesian_basis_sets = Some true,
        },
        system = exess_rex::System {
          oversubscribe_gpus = None,
          teams_per_node = None,
          gpus_per_team = None,
          max_gpu_memory_mb = None,
        },
        keywords = exess_rex::Keywords {
          scf = None,
          ks = None,
          rtat = None,
          frag = Some (exess_rex::FragKeywords {
            cutoffs = Some (exess_rex::FragmentCutoffs {
              dimer = Some $dimer_cutoff,
              trimer = Some $trimer_cutoff,
              tetramer = Some $tetramer_cutoff,
              pentamer = None,
              hexamer = None, 
              heptamer = None,
              octamer = None,
            }),
            cutoff_type = None,
            distance_metric = None,
            level = exess_rex::FragmentLevel::$fragment_level,
            included_fragments = None,
            reference_fragment = None,
            enable_speed = None,
          }),
          boundary = None,
          log = None,
          dynamics = None,
          integrals = None,
          debug = None,
          export = None,
          guess = None,
          force_field = None,
          optimization = None,
          hessian = None,
          gradient = None,
          qmmm = None,
        },
        driver = exess_rex::Driver::Energy,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        runspec=runspec.substitute(
            target=f"Some ModuleInstanceTarget::{target}" if target else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        basis=basis,
        aux_basis=aux_basis,
        method=method,
        fragment_level=fragment_level,
        dimer_cutoff=dimer_cutoff,
        trimer_cutoff=trimer_cutoff,
        tetramer_cutoff=tetramer_cutoff,
    )
    result = None
    try:
        result = submit_rex(PROJECT_ID, rex)
        print(f"Run ID: {result['id']}")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"][0]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            with open(f"{qm_output_vobj['path']}.json", "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            return qm_output_json["qmmbe"]["expanded_hf_energy"]
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def interaction_energy(
    topology_path: Path,
    reference_fragment: int,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT = "cc-pVDZ-RIFIT",
    fragment_level: FragmentLevelT = "Dimer",
    dimer_cutoff: float = 100.0,
    trimer_cutoff: float = 25.0,
    tetramer_cutoff: float = 10.0,
    target: Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"] | None = None,
):
    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_rex_s
      ($runspec)
      (exess_rex::ExessParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = exess_rex::Model {
          method = exess_rex::Method::$method,
          basis = "$basis",
          aux_basis = Some "$aux_basis",
          standard_orientation = Some exess_rex::StandardOrientation::FullSystem,
          force_cartesian_basis_sets = Some true,
        },
        system = exess_rex::System {
          oversubscribe_gpus = None,
          teams_per_node = None,
          gpus_per_team = None,
          max_gpu_memory_mb = None,
        },
        keywords = exess_rex::Keywords {
          scf = None,
          ks = None,
          rtat = None,
          frag = Some (exess_rex::FragKeywords {
            cutoffs = Some (exess_rex::FragmentCutoffs {
              dimer = Some $dimer_cutoff,
              trimer = Some $trimer_cutoff,
              tetramer = Some $tetramer_cutoff,
              pentamer = None,
              hexamer = None, 
              heptamer = None,
              octamer = None,
            }),
            cutoff_type = None,
            distance_metric = None,
            level = exess_rex::FragmentLevel::$fragment_level,
            included_fragments = None,
            reference_fragment = Some $reference_fragment,
            enable_speed = None,
          }),
          boundary = None,
          log = None,
          dynamics = None,
          integrals = None,
          debug = None,
          export = None,
          guess = None,
          force_field = None,
          optimization = None,
          hessian = None,
          gradient = None,
          qmmm = None,
        },
        driver = exess_rex::Driver::Energy,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        runspec=runspec.substitute(
            target=f"Some ModuleInstanceTarget::{target}" if target else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        reference_fragment=reference_fragment,
        basis=basis,
        aux_basis=aux_basis,
        method=method,
        fragment_level=fragment_level,
        dimer_cutoff=dimer_cutoff,
        trimer_cutoff=trimer_cutoff,
        tetramer_cutoff=tetramer_cutoff,
    )
    result = None
    try:
        result = submit_rex(PROJECT_ID, rex)
        print(f"Run ID: {result['id']}")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"][0]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            with open(f"{qm_output_vobj['path']}.json", "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            return qm_output_json["qmmbe"]["expanded_hf_energy"]
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def chelpg(
    topology_path: Path,
    target: Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"] | None = None,
):
    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_rex_s
      ($runspec)
      (exess_rex::ExessParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = exess_rex::Model {
          method = exess_rex::Method::RestrictedHF,
          basis = "cc-pVDZ",
          aux_basis = None,
          standard_orientation = Some exess_rex::StandardOrientation::FullSystem,
          force_cartesian_basis_sets = Some true,
        },
        system = exess_rex::System {
          oversubscribe_gpus = None,
          teams_per_node = None,
          gpus_per_team = None,
          max_gpu_memory_mb = None,
        },
        keywords = exess_rex::Keywords {
          scf = Some (exess_rex::SCFKeywords {
            max_iters = Some 50,
            max_diis_history_length = Some 12,
            batch_size = Some 2560,
            convergence_metric = Some exess_rex::ConvergenceMetric::DIIS,
            convergence_threshold = Some 0.00000001,
            density_threshold = Some 0.0000000001,
            gradient_screening_threshold = Some 0.0000000001,
            bf_cutoff_threshold = None,
            density_basis_set_projection_fallback_enabled = None,
            use_ri = Some false,
            store_ri_b_on_host = Some false,
            compress_ri_b = Some false,
            homo_lumo_guess_rotation_angle = None,
            fock_build_type = Some exess_rex::FockBuildType::HGP,
            exchange_screening_threshold = Some 0.00001,
            group_shared_exponents = Some false,
          }),
          ks = None,
          rtat = None,
          frag = Some (exess_rex::FragKeywords {
            cutoffs = None,
            cutoff_type = None,
            distance_metric = None,
            level = exess_rex::FragmentLevel::Monomer,
            included_fragments = None,
            reference_fragment = None,
            enable_speed = None,
          }),
          boundary = None,
          log = None,
          dynamics = None,
          integrals = None,
          debug = None,
          export = Some (exess_rex::ExportKeywords {
            export_density = None,
            export_relaxed_mp2_density_correction = None,
            export_fock = None,
            export_overlap = None,
            export_h_core = None,
            export_expanded_density = None,
            export_expanded_gradient = None,
            export_molecular_orbital_coeffs = None,
            export_gradient = None,
            export_external_charge_gradient = None,
            export_mulliken_charges = None,
            export_chelpg_charges = Some true,
            export_bond_orders = Some true,
            export_h_caps = None,
            export_density_descriptors = None,
            export_esp_descriptors = None,
            export_expanded_esp_descriptors = None,
            export_basis_labels = None,
            export_hessian = None,
            export_mass_weighted_hessian = None,
            export_hessian_frequencies = None,
            flatten_symmetric = None,
            concatenate_hdf5_files = None,
            light_json = None,
            descriptor_grid = None,
          }),
          guess = None,
          force_field = None,
          optimization = None,
          hessian = None,
          gradient = None,
          qmmm = None,
        },
        driver = exess_rex::Driver::Energy,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        runspec=runspec.substitute(
            target=f"Some ModuleInstanceTarget::{target}" if target else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
    )
    result = None
    try:
        result = submit_rex(PROJECT_ID, rex)
        print(f"Run ID: {result['id']}")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"][0]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            with open(f"{qm_output_vobj['path']}.json", "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            qm_output_vobj = result["result"]["Ok"][1]
            qm_output = download_object(qm_output_vobj["path"])
            decompressed = zstd.ZstdDecompressor().decompress(
                qm_output, max_output_size=int(1e8)
            )
            with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
                hdf5_f = tar.extractfile(tar.getnames()[1])
                with h5py.File(hdf5_f, "r") as f:
                    chelpg = [float(x) for x in f["monomers/0/chelpg_charges"]]  # pyright: ignore[reportGeneralTypeIssues]
            return chelpg
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def qmmm(
    topology_path: Path,
    n_timesteps: int,
    dt_ps: float = 0.002,
    temperature_kelvin: float = 290.0,
    pressure_atm: float | None = None,
    qm_fragments: list[int] | None = None,
    err_tol_kj_per_mol_nm: float = 10.0,
    max_iterations: int = 0,
    trajectory_format: Literal["XYZ", "xyz", "JSON", "json"] | None = None,
    interval: int | None = None,
    start: int | None = None,
    end: int | None = None,
    include_waters: int | None = None,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT = "cc-pVDZ-RIFIT",
    fragment_level: FragmentLevelT = "Monomer",
    dimer_cutoff: float = 100.0,
    trimer_cutoff: float = 25.0,
    tetramer_cutoff: float = 10.0,
    target: Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"] | None = None,
):
    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_rex_s
      ($runspec)
      (exess_rex::ExessParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = exess_rex::Model {
          method = exess_rex::Method::$method,
          basis = "$basis",
          aux_basis = Some "$aux_basis",
          standard_orientation = Some exess_rex::StandardOrientation::FullSystem,
          force_cartesian_basis_sets = Some true,
        },
        system = exess_rex::System {
          oversubscribe_gpus = None,
          teams_per_node = None,
          gpus_per_team = None,
          max_gpu_memory_mb = None,
        },
        keywords = exess_rex::Keywords {
          scf = None,
          ks = None,
          rtat = None,
          frag = Some (exess_rex::FragKeywords {
            cutoffs = Some (exess_rex::FragmentCutoffs {
              dimer = Some $dimer_cutoff,
              trimer = Some $trimer_cutoff,
              tetramer = Some $tetramer_cutoff,
              pentamer = None,
              hexamer = None, 
              heptamer = None,
              octamer = None,
            }),
            cutoff_type = None,
            distance_metric = None,
            level = exess_rex::FragmentLevel::$fragment_level,
            included_fragments = None,
            reference_fragment = None,
            enable_speed = None,
          }),
          boundary = None,
          log = None,
          dynamics = None,
          integrals = None,
          debug = None,
          export = None,
          guess = None,
          force_field = None,
          optimization = None,
          hessian = None,
          gradient = None,
          qmmm = Some (exess_rex::QMMMKeywords {
            n_timesteps = $n_timesteps,
            dt_ps = $dt_ps,
            temperature_kelvin = $temperature_kelvin,
            pressure_atm = $maybe_pressure_atm,
            qm_fragments = $qm_fragments,
            minimisation = Some (exess_rex::ClassicalMinimisation {
              err_tol_kj_per_mol_nm = $err_tol_kj_per_mol_nm,
              max_iterations = $max_iterations,
            }),
            trajectory = Some (exess_rex::MDTrajectory {
              format = $maybe_format,
              interval = $maybe_interval,
              start = $maybe_start,
              end = $maybe_end,
              include_waters = $maybe_include_waters,
            }),
          }),
        },
        driver = exess_rex::Driver::Dynamics,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        runspec=runspec.substitute(
            target=f"Some ModuleInstanceTarget::{target}" if target else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        basis=basis,
        aux_basis=aux_basis,
        method=method,
        fragment_level=fragment_level,
        dimer_cutoff=dimer_cutoff,
        trimer_cutoff=trimer_cutoff,
        tetramer_cutoff=tetramer_cutoff,
        n_timesteps=n_timesteps,
        dt_ps=dt_ps,
        temperature_kelvin=temperature_kelvin,
        maybe_pressure_atm=optional_str(pressure_atm),
        qm_fragments=qm_fragments or [],
        err_tol_kj_per_mol_nm=err_tol_kj_per_mol_nm,
        max_iterations=max_iterations,
        maybe_format=optional_str(trajectory_format),
        maybe_interval=optional_str(interval),
        maybe_start=optional_str(start),
        maybe_end=optional_str(end),
        maybe_include_waters=optional_str(include_waters),
    )
    result = None
    try:
        result = submit_rex(PROJECT_ID, rex)
        print(f"Run ID: {result['id']}")
        print(f"Status: {result['status']}")
        print(f"Result: {result['result']}")
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"][0]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            with open(f"{qm_output_vobj['path']}.json", "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            return qm_output_json
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def run_energy():
    cyclopts.run(energy)


def run_interaction_energy():
    cyclopts.run(interaction_energy)


def run_chelpg():
    cyclopts.run(chelpg)


def run_qmmm():
    cyclopts.run(qmmm)


if __name__ == "__main__":
    i_folder = Path.cwd() / ".." / "libqdx" / ".scratch" / "qm-affinity" / "i"

    # o = energy(i_folder / "thrombin_1c_t.json")
    # o = interaction_energy(i_folder / "tyk2_ejm_31_t.json", 1)
    # o = chelpg(i_folder / "tyk2_ejm_31_t.json")
    o = qmmm(i_folder / "tyk2_ejm_31_t.json", n_timesteps=100)

    print(o)

# TODO:
#  - trace for failure
#  - stdout, stderr
#  - other module instance info?
