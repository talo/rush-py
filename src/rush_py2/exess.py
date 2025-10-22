#!/usr/bin/env python3
import json
import tarfile
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Literal

import cyclopts
import h5py
import zstandard as zstd
from gql.transport.exceptions import TransportQueryError

from rush_py2.client import (
    PROJECT_ID,
    download_object,
    print_run_trace,
    runspec,
    submit_rex,
    upload_object,
)

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

type GradientMethodT = Literal["Analytical", "Numerical"]

type TargetT = Literal["Bullet", "Bullet2", "Bullet3", "Gadi", "Setonix"]


def clean_dict(d):
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d]
    else:
        return d


def optional_str(v: str | int | float | list[int] | bool | None) -> str:
    return f"Some {v}" if v is not None else "None"


def energy(
    topology_path: Path | str,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT = "cc-pVDZ-RIFIT",
    fragment_level: FragmentLevelT = "Monomer",
    dimer_cutoff: float = 100.0,
    trimer_cutoff: float = 25.0,
    tetramer_cutoff: float = 10.0,
    target: TargetT | None = None,
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
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"][0]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            out_path = f"{qm_output_vobj['path']}.json"
            with open(out_path, "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            return out_path
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")
        elif result["status"] == "error":
            print_run_trace(result)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def interaction_energy(
    topology_path: Path | str,
    reference_fragment: int,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT = "cc-pVDZ-RIFIT",
    fragment_level: FragmentLevelT = "Dimer",
    dimer_cutoff: float = 100.0,
    trimer_cutoff: float = 25.0,
    tetramer_cutoff: float = 10.0,
    target: TargetT | None = None,
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
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"][0]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            out_path = f"{qm_output_vobj['path']}.json"
            with open(out_path, "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            return out_path
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")
        elif result["status"] == "error":
            print_run_trace(result)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def chelpg(
    topology_path: Path | str,
    target: TargetT | None = None,
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
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"][0]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            out_path = f"{qm_output_vobj['path']}.json"
            with open(out_path, "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            qm_output_vobj = result["result"]["Ok"][1]
            qm_output = download_object(qm_output_vobj["path"])
            decompressed = zstd.ZstdDecompressor().decompress(
                qm_output, max_output_size=int(1e8)
            )
            with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
                hdf5_f = tar.extractfile(tar.getnames()[1])
                chelpg = []
                with h5py.File(hdf5_f, "r") as f:
                    frag_indices = [int(x) for x in f["monomers"].keys()]
                    for frag_idx in sorted(frag_indices):
                        # pyright: ignore[reportGeneralTypeIssues]
                        chelpg += [
                            float(x) for x in f[f"monomers/{frag_idx}/chelpg_charges"]
                        ]
            return (out_path, chelpg)
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")
        elif result["status"] == "error":
            print_run_trace(result)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def qmmm(
    topology_path: Path | str,
    residues_path: Path | str,
    n_timesteps: int,
    dt_ps: float = 0.002,
    gradient_finite_difference_step_size: float | None = None,
    temperature_kelvin: float = 290.0,
    pressure_atm: float | None = None,
    qm_fragments: list[int] | None = None,
    err_tol_kj_per_mol_nm: float = 10.0,
    max_iterations: int = 0,
    interval: int | None = None,
    start: int | None = None,
    end: int | None = None,
    include_waters: int | None = None,
    fixed_atoms: list[int] | None = None,
    free_atoms: list[int] | None = None,
    fixed_fragments: list[int] | None = None,
    free_fragments: list[int] | None = None,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "STO-3G",
    aux_basis: AuxBasisT | None = None,
    fragment_level: FragmentLevelT = "Monomer",
    dimer_cutoff: float = 100.0,
    trimer_cutoff: float = 25.0,
    tetramer_cutoff: float = 10.0,
    target: TargetT | None = None,
):
    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)
    residues_vobj = upload_object(PROJECT_ID, residues_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology residues →
    exess_qmmm_rex_s
      ($runspec)
      (exess_qmmm_rex::QMMMParams {
        schema_version = "0.2.0",
        model = Some (exess_qmmm_rex::Model {
          method = exess_qmmm_rex::Method::$method,
          basis = "$basis",
          aux_basis = $maybe_aux_basis,
          standard_orientation = Some exess_qmmm_rex::StandardOrientation::FullSystem,
          force_cartesian_basis_sets = Some true,
        }),
        system = Some (exess_qmmm_rex::System {
          oversubscribe_gpus = None,
          teams_per_node = None,
          gpus_per_team = None,
          max_gpu_memory_mb = None,
        }),
        keywords = exess_qmmm_rex::Keywords {
          scf = None,
          ks = None,
          rtat = None,
          frag = Some (exess_qmmm_rex::FragKeywords {
            cutoffs = Some (exess_qmmm_rex::FragmentCutoffs {
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
            level = exess_qmmm_rex::FragmentLevel::$fragment_level,
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
          gradient = Some (exess_qmmm_rex::GradientKeywords {
            finite_difference_step_size = $maybe_gradient_finite_difference_step_size,
            method = Some exess_qmmm_rex::DerivativesMethod::Analytical,
          }),
          qmmm = Some (exess_qmmm_rex::QMMMKeywords {
            n_timesteps = $n_timesteps,
            dt_ps = $dt_ps,
            temperature_kelvin = $temperature_kelvin,
            pressure_atm = $maybe_pressure_atm,
            qm_fragments = $qm_fragments,
            minimisation = None,
            trajectory = Some (exess_qmmm_rex::MDTrajectory {
              format = None,
              interval = $maybe_interval,
              start = $maybe_start,
              end = $maybe_end,
              include_waters = $maybe_include_waters,
            }),
            restraints = Some (exess_qmmm_rex::MDRestraints {
              fixed_atoms = $maybe_fixed_atoms,
              free_atoms = $maybe_free_atoms,
              fixed_fragments = $maybe_fixed_fragments,
              free_fragments = $maybe_free_fragments,
              fix_heavy = None,
            }),
          }),
        },
      })
      (obj_j topology)
      (Some (obj_j residues))
in
  exess "$topology_vobj_path" "$residues_vobj_path"
""").substitute(
        runspec=runspec.substitute(
            target=f"Some ModuleInstanceTarget::{target}" if target else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"],
        basis=basis,
        maybe_aux_basis=optional_str(aux_basis),
        method=method,
        fragment_level=fragment_level,
        dimer_cutoff=dimer_cutoff,
        trimer_cutoff=trimer_cutoff,
        tetramer_cutoff=tetramer_cutoff,
        maybe_gradient_finite_difference_step_size=optional_str(
            gradient_finite_difference_step_size
        ),
        n_timesteps=n_timesteps,
        dt_ps=dt_ps,
        temperature_kelvin=temperature_kelvin,
        maybe_pressure_atm=optional_str(pressure_atm),
        qm_fragments=qm_fragments if qm_fragments is not None else [],
        err_tol_kj_per_mol_nm=err_tol_kj_per_mol_nm,
        max_iterations=max_iterations,
        maybe_interval=optional_str(interval),
        maybe_start=optional_str(start),
        maybe_end=optional_str(end),
        maybe_include_waters=optional_str(include_waters),
        maybe_fixed_atoms=optional_str(fixed_atoms),
        maybe_free_atoms=optional_str(free_atoms),
        maybe_fixed_fragments=optional_str(fixed_fragments),
        maybe_free_fragments=optional_str(free_fragments),
    )
    result = None
    try:
        result = submit_rex(PROJECT_ID, rex)
        if "Ok" in result["result"]:
            qm_output_vobj = result["result"]["Ok"]
            qm_output_json = json.loads(
                download_object(qm_output_vobj["path"]).decode()
            )
            out_path = f"{qm_output_vobj['path']}.json"
            with open(out_path, "w") as f:
                json.dump(clean_dict(qm_output_json), f, indent=2)
            return out_path
        elif "Err" in result["result"]:
            print(f"Error: {result['result']['Err']}")
        elif result["status"] == "error":
            print_run_trace(result)

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
    o = qmmm(
        "6a5j_t.json",
        "6a5j_r.json",
        n_timesteps=504,
        qm_fragments=[],
        free_atoms=[0],
    )
    print(f"Output: {o}")

# TODO:
#  - trace for failure
#  - stdout, stderr
#  - other module instance info?
#  - qmmm minimisation config:
#    minimisation = Some (exess_rex::ClassicalMinimisation {
#      err_tol_kj_per_mol_nm = $err_tol_kj_per_mol_nm,
#      max_iterations = $max_iterations,
#    }),
