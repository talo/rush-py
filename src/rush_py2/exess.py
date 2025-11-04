#!/usr/bin/env python3
import json
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Literal

import cyclopts
import h5py
import zstandard as zstd
from gql.transport.exceptions import TransportQueryError

from .client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    collect_run,
    download_object,
    print_run_trace,
    submit_rex,
    upload_object,
)
from .utils import bool_to_str, clean_dict, float_to_str, optional_str

type MethodT = Literal[
    "RestrictedHF",
    "UnrestrictedHF",
    "RestrictedKSDFT",
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

type ConvergenceMetricT = Literal["Energy", "DIIS", "Density"]

type FockBuildTypeT = Literal["HGP", "UM09", "RI"]

type FragmentLevelT = Literal[
    "Monomer",
    "Dimer",
    "Trimer",
    "Tetramer",
]

type CutoffTypeT = Literal["Centroid", "ClosestPair"]


@dataclass
class SCFKeywords:
    max_iters: int = 50
    max_diis_history_length: int = 8
    batch_size: int = 2560
    convergence_metric: ConvergenceMetricT = "DIIS"
    convergence_threshold: float = 1e-6
    density_threshold: float = 1e-10
    gradient_screening_threshold: float = 1e-10
    bf_cutoff_threshold: float | None = None
    density_basis_set_projection_fallback_enabled: bool | None = None
    use_ri: bool = False
    store_ri_b_on_host: bool = False
    compress_ri_b: bool = False
    homo_lumo_guess_rotation_angle: float | None = None
    fock_build_type: FockBuildTypeT = "HGP"
    exchange_screening_threshold: float = 1e-5
    group_shared_exponents: bool = False

    def to_rex(self):
        return Template(
            """Some (exess_rex::SCFKeywords {
            max_iters = Some $max_iters,
            max_diis_history_length = Some $max_diis_history_length,
            batch_size = Some $batch_size,
            convergence_metric = Some exess_rex::ConvergenceMetric::$convergence_metric,
            convergence_threshold = Some $convergence_threshold,
            density_threshold = Some $density_threshold,
            gradient_screening_threshold = Some $gradient_screening_threshold,
            bf_cutoff_threshold = $maybe_bf_cutoff_threshold,
            density_basis_set_projection_fallback_enabled = $maybe_density_basis_set_projection_fallback_enabled,
            use_ri = Some $use_ri,
            store_ri_b_on_host = Some $store_ri_b_on_host,
            compress_ri_b = Some $compress_ri_b,
            homo_lumo_guess_rotation_angle = $maybe_homo_lumo_guess_rotation_angle,
            fock_build_type = Some exess_rex::FockBuildType::$fock_build_type,
            exchange_screening_threshold = Some $exchange_screening_threshold,
            group_shared_exponents = Some $group_shared_exponents,
          })"""
        ).substitute(
            max_iters=self.max_iters,
            max_diis_history_length=self.max_diis_history_length,
            batch_size=self.batch_size,
            convergence_metric=self.convergence_metric,
            convergence_threshold=float_to_str(self.convergence_threshold),
            density_threshold=float_to_str(self.density_threshold),
            gradient_screening_threshold=float_to_str(
                self.gradient_screening_threshold
            ),
            maybe_bf_cutoff_threshold=optional_str(self.bf_cutoff_threshold),
            maybe_density_basis_set_projection_fallback_enabled=optional_str(
                self.density_basis_set_projection_fallback_enabled
            ),
            use_ri=bool_to_str(self.use_ri),
            store_ri_b_on_host=bool_to_str(self.store_ri_b_on_host),
            compress_ri_b=bool_to_str(self.compress_ri_b),
            maybe_homo_lumo_guess_rotation_angle=optional_str(
                self.homo_lumo_guess_rotation_angle
            ),
            fock_build_type=self.fock_build_type,
            exchange_screening_threshold=float_to_str(
                self.exchange_screening_threshold
            ),
            group_shared_exponents=bool_to_str(self.group_shared_exponents),
        )


@dataclass
class FragKeywords:
    """
    Configure the fragmentation of the system.

    Defaults are provided for all relevant levels.
    NOTE: cutoffs for each level must be less than or equal to those at the lower levels.
    """

    level: FragmentLevelT = "Dimer"
    dimer_cutoff: float | None = None
    trimer_cutoff: float | None = None
    tetramer_cutoff: float | None = None
    cutoff_type: CutoffTypeT | None = None

    def __post_init__(self):
        if self.level == "Monomer":
            self.dimer_cutoff = 100.0
            self.trimer_cutoff = None
            self.tetramer_cutoff = None
            self.cutoff_type = None
        if self.level == "Dimer" and self.dimer_cutoff is None:
            self.dimer_cutoff = 100.0
            self.trimer_cutoff = None
            self.tetramer_cutoff = None
        if self.level == "Trimer":
            if self.dimer_cutoff is None:
                self.dimer_cutoff = 100.0
            if self.trimer_cutoff is None:
                self.trimer_cutoff = 25.0
            self.tetramer_cutoff = None
        if self.level == "Tetramer":
            if self.dimer_cutoff is None:
                self.dimer_cutoff = 100.0
            if self.trimer_cutoff is None:
                self.trimer_cutoff = 25.0
            if self.tetramer_cutoff is None:
                self.tetramer_cutoff = 10.0

    def to_rex(self, reference_fragment: int | None = None):
        return Template(
            """Some (exess_rex::FragKeywords {
            cutoffs = Some (exess_rex::FragmentCutoffs {
              dimer = $dimer_cutoff,
              trimer = $trimer_cutoff,
              tetramer = $tetramer_cutoff,
              pentamer = None,
              hexamer = None,
              heptamer = None,
              octamer = None,
            }),
            cutoff_type = $maybe_cutoff_type,
            distance_metric = None,
            level = exess_rex::FragmentLevel::$level,
            included_fragments = None,
            reference_fragment = $maybe_reference_fragment,
            enable_speed = None,
          })"""
        ).substitute(
            dimer_cutoff=optional_str(self.dimer_cutoff),
            trimer_cutoff=optional_str(self.trimer_cutoff),
            tetramer_cutoff=optional_str(self.tetramer_cutoff),
            maybe_cutoff_type=optional_str(
                self.cutoff_type, "exess_rex::FragmentDistanceMethod::"
            ),
            level=self.level,
            maybe_reference_fragment=optional_str(reference_fragment),
        )


@dataclass
class Trajectory:
    """
    Configure the output of QMMM runs. By default, will provide all atoms at every frame.
    """

    interval: int | None = None
    start: int | None = None
    end: int | None = None
    include_waters: int | None = None

    def to_rex(self):
        return Template(
            """Some (exess_qmmm_rex::MDTrajectory {
              format = None,
              interval = $maybe_interval,
              start = $maybe_start,
              end = $maybe_end,
              include_waters = $maybe_include_waters,
            })"""
        ).substitute(
            maybe_interval=optional_str(self.interval),
            maybe_start=optional_str(self.start),
            maybe_end=optional_str(self.end),
            maybe_include_waters=optional_str(self.include_waters),
        )


@dataclass
class Restraints:
    """
    Restrain atoms using an external force proportional to its distance from its original position,
    scaled by `k` (larger values mean a stronger restraint).

    All atoms can be fixed by specifying `free_atoms = []`.
    """

    k: float | None = None
    fixed_atoms: list[int] | None = None
    free_atoms: list[int] | None = None
    fixed_fragments: list[int] | None = None
    free_fragments: list[int] | None = None
    fix_heavy: bool | None = None

    def to_rex(self):
        return Template(
            """Some (exess_rex::Restraints {
              k = $maybe_k,
              fixed_atoms = $maybe_fixed_atoms,
              free_atoms = $maybe_free_atoms,
              fixed_fragments = $maybe_fixed_fragments,
              free_fragments = $maybe_free_fragments,
              fix_heavy = $maybe_fix_heavy,
            })"""
        ).substitute(
            maybe_k=optional_str(self.k),
            maybe_fixed_atoms=optional_str(self.fixed_atoms),
            maybe_free_atoms=optional_str(self.free_atoms),
            maybe_fixed_fragments=optional_str(self.fixed_fragments),
            maybe_free_fragments=optional_str(self.free_fragments),
            maybe_fix_heavy=optional_str(self.fix_heavy),
        )


def collect_energy(run_id: str):
    run = collect_run(run_id)
    if "Ok" in run["result"]:
        qm_output_vobj = run["result"]["Ok"][0]
        qm_output_json = json.loads(download_object(qm_output_vobj["path"]).decode())
        out_path = f"{qm_output_vobj['path']}.json"
        with open(out_path, "w") as f:
            json.dump(clean_dict(qm_output_json), f, indent=2)
        return out_path
    elif "Err" in run["result"]:
        print(f"Error: {run['result']['Err']}")
    elif run["status"] == "error":
        print_run_trace(run)


def energy(
    topology_path: Path | str,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords = FragKeywords(),
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the energy of the system in the QDX topology file at `topology_path`.
    """

    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_rex_s
      ($run_spec)
      (exess_rex::ExessParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = Some (exess_rex::Model {
          method = exess_rex::Method::$method,
          basis = "$basis",
          aux_basis = $maybe_aux_basis,
          standard_orientation = None,
          force_cartesian_basis_sets = None,
        }),
        system = None,
        keywords = exess_rex::Keywords {
          scf = $scf_keywords,
          ks = None,
          rtat = None,
          frag = $frag_keywords,
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
          machine_learning = None,
          regions = None,
        },
        driver = exess_rex::Driver::Energy,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        run_spec=run_spec.to_rex(),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(f'"{aux_basis}"'),
        scf_keywords=scf_keywords.to_rex() if scf_keywords is not None else "None",
        frag_keywords=frag_keywords.to_rex() if frag_keywords is not None else "None",
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        run_id = submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_energy(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def interaction_energy(
    topology_path: Path | str,
    reference_fragment: int,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords = FragKeywords(),
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the interaction energy between the fragment with index `reference_fragment` and the rest of the system
    in the toplogy file at `topology_path`.
    """

    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_rex_s
      ($run_spec)
      (exess_rex::ExessParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = Some (exess_rex::Model {
          method = exess_rex::Method::$method,
          basis = "$basis",
          aux_basis = $maybe_aux_basis,
          standard_orientation = None,
          force_cartesian_basis_sets = None,
        }),
        system = None,
        keywords = exess_rex::Keywords {
          scf = $scf_keywords,
          ks = None,
          rtat = None,
          frag = $frag_keywords,
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
          machine_learning = None,
          regions = None,
        },
        driver = exess_rex::Driver::Energy,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        run_spec=run_spec.to_rex(),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(f'"{aux_basis}"'),
        scf_keywords=scf_keywords.to_rex() if scf_keywords is not None else "None",
        frag_keywords=frag_keywords.to_rex(reference_fragment),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        run_id = submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_energy(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def chelpg(
    topology_path: Path | str,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the CHELPG partial charges for all atoms of the system in the topology file at `topology_path`.
    """

    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_rex_s
      ($run_spec)
      (exess_rex::ExessParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = Some (exess_rex::Model {
          method = exess_rex::Method::RestrictedHF,
          basis = "cc-pVDZ",
          aux_basis = None,
          standard_orientation = None,
          force_cartesian_basis_sets = None,
        }),
        system = None,
        keywords = exess_rex::Keywords {
          scf = $scf_keywords,
          ks = None,
          rtat = None,
          frag = $frag_keywords,
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
            light_json = None,
            concatenate_hdf5_files = None,
            training_db = None,
            descriptor_grid = None,
          }),
          guess = None,
          force_field = None,
          optimization = None,
          hessian = None,
          gradient = None,
          qmmm = None,
          machine_learning = None,
          regions = None,
        },
        driver = exess_rex::Driver::Energy,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        run_spec=run_spec.to_rex(),
        scf_keywords=SCFKeywords(
            max_diis_history_length=12, convergence_threshold=1e-8
        ).to_rex(),
        frag_keywords=FragKeywords(level="Monomer").to_rex(),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        run_id = submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            run = collect_run(run_id)
            if "Ok" in run["result"]:
                qm_output_vobj = run["result"]["Ok"][0]
                qm_output_json = json.loads(
                    download_object(qm_output_vobj["path"]).decode()
                )
                out_path = f"{qm_output_vobj['path']}.json"
                with open(out_path, "w") as f:
                    json.dump(clean_dict(qm_output_json), f, indent=2)
                qm_output_vobj = run["result"]["Ok"][1]
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
                                float(x)
                                for x in f[f"monomers/{frag_idx}/chelpg_charges"]
                            ]
                return (out_path, chelpg)
            elif "Err" in run["result"]:
                print(f"Error: {run['result']['Err']}")
            elif run["status"] == "error":
                print_run_trace(run)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


def qmmm(
    topology_path: Path | str,
    residues_path: Path | str,
    n_timesteps: int,
    dt_ps: float = 2e-3,
    temperature_kelvin: float = 290.0,
    pressure_atm: float | None = None,
    restraints: Restraints | None = None,
    trajectory: Trajectory = Trajectory(),
    gradient_finite_difference_step_size: float | None = None,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "STO-3G",
    aux_basis: AuxBasisT | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords = FragKeywords(),
    qm_fragments: list[int] | None = None,
    mm_fragments: list[int] | None = None,
    ml_fragments: list[int] | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run a QMMM simulation of the system in the QDX topology and residues files at `topology_path` and `residues_path`.

    Specifying the numberof timesteps is mandatory.
    If pressure is None, an NVT ensemble is used; if pressure is specified, an NPT ensemble is used.
    Fragments can be specified as QM, MM, or ML fragments via the respective parameters.
    If two fragment list parameters are specified, the rest of the fragments are inferred to be of the other type.
    If three fragment list parameters are specified, each fragment must be placed in exactly one of the lists.
    It is invalid to specify one fragment list parameter.
    """

    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)
    residues_vobj = upload_object(PROJECT_ID, residues_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology residues →
    exess_qmmm_rex_s
      ($run_spec)
      (exess_qmmm_rex::QMMMParams {
        schema_version = "0.2.0",
        model = Some (exess_qmmm_rex::Model {
          method = exess_qmmm_rex::Method::$method,
          basis = "$basis",
          aux_basis = $maybe_aux_basis,
          standard_orientation = None,
          force_cartesian_basis_sets = None,
        }),
        system = None,
        keywords = exess_qmmm_rex::Keywords {
          scf = $scf_keywords,
          ks = None,
          rtat = None,
          frag = $frag_keywords,
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
            minimisation = None,
            trajectory = $trajectory,
            restraints = $restraints,
            energy_csv = None,
          }),
          machine_learning = None,
          regions = Some (exess_qmmm_rex::RegionKeywords {
            qm_fragments = $maybe_qm_fragments,
            mm_fragments = $maybe_mm_fragments,
            ml_fragments = $maybe_ml_fragments,
          }),
        },
      })
      (obj_j topology)
      (Some (obj_j residues))
in
  exess "$topology_vobj_path" "$residues_vobj_path"
""").substitute(
        run_spec=run_spec.to_rex(),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(f'"{aux_basis}"'),
        scf_keywords=scf_keywords.to_rex() if scf_keywords is not None else "None",
        frag_keywords=frag_keywords.to_rex() if frag_keywords is not None else "None",
        maybe_gradient_finite_difference_step_size=optional_str(
            gradient_finite_difference_step_size
        ),
        n_timesteps=n_timesteps,
        dt_ps=dt_ps,
        temperature_kelvin=temperature_kelvin,
        maybe_pressure_atm=optional_str(pressure_atm),
        trajectory=trajectory.to_rex(),
        restraints=restraints.to_rex() if restraints is not None else "None",
        maybe_qm_fragments=optional_str(qm_fragments),
        maybe_mm_fragments=optional_str(mm_fragments),
        maybe_ml_fragments=optional_str(ml_fragments),
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"],
    )
    try:
        run_id = submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            run = collect_run(run_id)
            if "Ok" in run["result"]:
                qm_output_vobj = run["result"]["Ok"]
                qm_output_json = json.loads(
                    download_object(qm_output_vobj["path"]).decode()
                )
                out_path = f"{qm_output_vobj['path']}.json"
                with open(out_path, "w") as f:
                    json.dump(clean_dict(qm_output_json), f, indent=2)
                return out_path
            elif "Err" in run["result"]:
                print(f"Error: {run['result']['Err']}")
            elif run["status"] == "error":
                print_run_trace(run)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}")


@dataclass
class OptimizationConvergenceCriteria:
    metric: str | None = None
    gradient_threshold: float | None = None
    delta_energy_threshold: float | None = None
    step_component_threshold: float | None = None

    def to_rex(self, reference_fragment: int | None = None):
        return Template(
            """Some (exess_geo_opt_rex::OptimizationConvergenceCriteria {
            metric = $maybe_metric,
            gradient_threshold = $maybe_gradient_threshold,
            delta_energy_threshold = $maybe_delta_energy_threshold,
            step_component_threshold = $maybe_step_component_threshold,
          })"""
        ).substitute(
            maybe_metric=optional_str(self.metric),  # TODO: enum prefix
            maybe_gradient_threshold=optional_str(self.gradient_threshold),
            maybe_delta_energy_threshold=optional_str(self.delta_energy_threshold),
            maybe_step_component_threshold=optional_str(self.step_component_threshold),
        )


type CoordinateSystemT = Literal["Cartesian", "NaturalInternal", "DelocalisedInternal"]

type HessianGuessTypeT = Literal["Identity", "ScaledIdentity", "Schlegel", "Lindh"]

type OptimizationAlgorithmTypeT = Literal[
    "EigenvectorFollowing", "TrustRegionAugmentedHessian", "LBFGS"
]


@dataclass
class TrustRegionKeywords:
    initial_radius: float | None = None
    max_radius: float | None = None
    min_radius: float | None = None
    increase_factor: float | None = None
    decrease_factor: float | None = None
    constrict_factor: float | None = None
    increase_threshold: float | None = None
    decrease_threshold: float | None = None
    rejection_threshold: float | None = None

    def to_rex(self):
        return Template(
            """Some (exess_geo_opt_rex::TrustRegionKeywords {
            initial_radius = $maybe_initial_radius,
            max_radius = $maybe_max_radius,
            min_radius = $maybe_min_radius,
            increase_factor = $maybe_increase_factor,
            decrease_factor = $maybe_decrease_factor,
            constrict_factor = $maybe_constrict_factor,
            increase_threshold = $maybe_increase_threshold,
            decrease_threshold = $maybe_decrease_threshold,
            rejection_threshold = $maybe_rejection_threshold,
          })"""
        ).substitute(
            maybe_initial_radius=optional_str(self.initial_radius),
            maybe_max_radius=optional_str(self.max_radius),
            maybe_min_radius=optional_str(self.min_radius),
            maybe_increase_factor=optional_str(self.increase_factor),
            maybe_decrease_factor=optional_str(self.decrease_factor),
            maybe_constrict_factor=optional_str(self.constrict_factor),
            maybe_increase_threshold=optional_str(self.increase_threshold),
            maybe_decrease_threshold=optional_str(self.decrease_threshold),
            maybe_rejection_threshold=optional_str(self.rejection_threshold),
        )


type LBFGSLinesearchT = Literal[
    "MoreThuente", "BacktrackingArmijo", "BacktrackingWolfe", "BacktrackingStrongWolfe"
]


@dataclass
class LBFGSKeywords:
    linesearch: LBFGSLinesearchT | None = None
    n_corrections: int | None = None
    epsilon: float | None = None
    max_linesearch: int | None = None
    gtol: float | None = None

    def to_rex(self):
        return Template(
            """Some (exess_geo_opt_rex::LBFGSKeywords {
              linesearch = $maybe_linesearch,
              n_corrections = $maybe_n_corrections,
              epsilon = $maybe_epsilon,
              max_linesearch = $maybe_max_linesearch,
              gtol = $maybe_gtol,
            })"""
        ).substitute(
            maybe_linesearch=optional_str(
                self.linesearch, "exess_geo_opt_rex::LBFGSLinesearch::"
            ),
            maybe_n_corrections=optional_str(self.n_corrections),
            maybe_epsilon=optional_str(self.epsilon),
            maybe_max_linesearch=optional_str(self.max_linesearch),
            maybe_gtol=optional_str(self.gtol),
        )


@dataclass
class OptimizationKeywords:
    convergence_criteria: OptimizationConvergenceCriteria | None = None
    optimizer_reset_interval: int | None = None
    coordinate_system: CoordinateSystemT | None = None
    constraints: list[list[int]] | None = None
    hessian_guess: HessianGuessTypeT | None = None
    algorithm: OptimizationAlgorithmTypeT | None = None
    lbfgs_keywords: LBFGSKeywords | None = None
    frozen_distance_slippage_tolerance_angstroms: float | None = None
    frozen_angle_slippage_tolerance_degrees: float | None = None
    trust_region_keywords: TrustRegionKeywords | None = None
    fixed_atoms: list[int] | None = None
    free_atoms: list[int] | None = None
    fixed_fragments: list[int] | None = None
    free_fragments: list[int] | None = None
    fix_heavy: bool | None = None

    def to_rex(self, max_iters):
        return Template(
            """Some (exess_geo_opt_rex::OptimizationKeywords {
            max_iters = $max_iters,
            convergence_criteria = $maybe_convergence_criteria,
            optimizer_reset_interval = $maybe_optimizer_reset_interval,
            coordinate_system = $maybe_coordinate_system,
            constraints = $maybe_constraints,
            hessian_guess = $maybe_hessian_guess,
            algorithm = $maybe_algorithm,
            lbfgs_keywords = $maybe_lbfgs_keywords,
            frozen_distance_slippage_tolerance_angstroms = $maybe_frozen_distance_slippage_tolerance_angstroms,
            frozen_angle_slippage_tolerance_degrees = $maybe_frozen_angle_slippage_tolerance_degrees,
            trust_region_keywords = $maybe_trust_region_keywords,
            fixed_atoms = $maybe_fixed_atoms,
            free_atoms = $maybe_free_atoms,
            fixed_fragments = $maybe_fixed_fragments,
            free_fragments = $maybe_free_fragments,
            fix_heavy = $maybe_fix_heavy,
          })"""
        ).substitute(
            max_iters=max_iters,
            maybe_convergence_criteria=(
                self.convergence_criteria.to_rex()
                if self.convergence_criteria is not None
                else "None"
            ),
            maybe_optimizer_reset_interval=optional_str(self.optimizer_reset_interval),
            maybe_coordinate_system=optional_str(
                self.coordinate_system, "exess_geo_opt_rex::CoordinateSystem::"
            ),
            # maybe_constraints=optional_list(
            #     self.constraints,
            #     lambda constraint: f"vec![{', '.join(f'exess_geo_opt_rex::AtomRef ({atom})' for atom in constraint)}]",
            # ),
            maybe_constraints="None",  # TODO
            maybe_hessian_guess=optional_str(
                self.hessian_guess, "exess_geo_opt_rex::HessianGuessType::"
            ),
            maybe_algorithm=optional_str(
                self.algorithm, "exess_geo_opt_rex::OptimizationAlgorithmType::"
            ),
            maybe_lbfgs_keywords=(
                self.lbfgs_keywords.to_rex()
                if self.lbfgs_keywords is not None
                else "None"
            ),
            maybe_frozen_distance_slippage_tolerance_angstroms=optional_str(
                self.frozen_distance_slippage_tolerance_angstroms
            ),
            maybe_frozen_angle_slippage_tolerance_degrees=optional_str(
                self.frozen_angle_slippage_tolerance_degrees
            ),
            maybe_trust_region_keywords=(
                self.trust_region_keywords.to_rex()
                if self.trust_region_keywords is not None
                else "None"
            ),
            maybe_fixed_atoms=optional_str(self.fixed_atoms),
            maybe_free_atoms=optional_str(self.free_atoms),
            maybe_fixed_fragments=optional_str(self.fixed_fragments),
            maybe_free_fragments=optional_str(self.free_fragments),
            maybe_fix_heavy=optional_str(self.fix_heavy),
        )


def optimization(
    topology_path: Path | str,
    max_iters: int,
    optimization_keywords: OptimizationKeywords = OptimizationKeywords(),
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    scf_keywords: SCFKeywords | None = None,
    qm_fragments: list[int] | None = None,
    mm_fragments: list[int] | None = None,
    ml_fragments: list[int] | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run optimization on the system in the QDX topology and residues files at `topology_path`.

    Specifying the maximum iterations is mandatory.
    Fragment-based QM calculation is not supported, but fragments can be used for specifying regions as QM, MM, or ML.
    If two fragment list parameters are specified, the rest of the fragments are inferred to be of the other type.
    If three fragment list parameters are specified, each fragment must be placed in exactly one of the lists.
    It is invalid to specify one fragment list parameter.
    """

    # Upload inputs
    topology_vobj = upload_object(PROJECT_ID, topology_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology →
    exess_geo_opt_rex_s
      ($run_spec)
      (exess_geo_opt_rex::OptimizationParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = Some (exess_geo_opt_rex::Model {
          method = exess_geo_opt_rex::Method::$method,
          basis = "$basis",
          aux_basis = $maybe_aux_basis,
          standard_orientation = None,
          force_cartesian_basis_sets = None,
        }),
        system = None,
        keywords = exess_geo_opt_rex::Keywords {
          scf = $scf_keywords,
          ks = None,
          rtat = None,
          frag = None,
          boundary = None,
          log = None,
          dynamics = None,
          integrals = None,
          debug = None,
          export = None,
          guess = None,
          force_field = None,
          optimization = $optimization_keywords,
          hessian = None,
          gradient = None,
          qmmm = None,
          machine_learning = None,
          regions = Some (exess_geo_opt_rex::RegionKeywords {
            qm_fragments = $maybe_qm_fragments,
            mm_fragments = $maybe_mm_fragments,
            ml_fragments = $maybe_ml_fragments,
          }),
        },
      })
      [ (obj_j topology) ]
in
  exess "$topology_vobj_path"
""").substitute(
        run_spec=run_spec.to_rex(),
        optimization_keywords=(
            optimization_keywords.to_rex(max_iters)
            if optimization_keywords is not None
            else "None"
        ),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(f'"{aux_basis}"'),
        scf_keywords=scf_keywords.to_rex() if scf_keywords is not None else "None",
        maybe_qm_fragments=optional_str(qm_fragments),
        maybe_mm_fragments=optional_str(mm_fragments),
        maybe_ml_fragments=optional_str(ml_fragments),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        run_id = submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            run = collect_run(run_id)
            if "Ok" in run["result"]:
                qm_output_vobj = run["result"]["Ok"][0]
                qm_output_json = json.loads(
                    download_object(qm_output_vobj["path"]).decode()
                )
                out_path1 = f"{qm_output_vobj['path']}.json"
                with open(out_path1, "w") as f:
                    json.dump(clean_dict(qm_output_json), f, indent=2)
                qm_output_vobj = run["result"]["Ok"][1]
                qm_output_json = json.loads(
                    download_object(qm_output_vobj["path"]).decode()
                )
                out_path2 = f"{qm_output_vobj['path']}.json"
                with open(out_path2, "w") as f:
                    json.dump(clean_dict(qm_output_json), f, indent=2)
                return (out_path1, out_path2)
            elif "Err" in run["result"]:
                print(f"Error: {run['result']['Err']}")
            elif run["status"] == "error":
                print_run_trace(run)
        else:
            return run_id

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


def run_optimization():
    cyclopts.run(optimization)


# TODO:
#  - trace for failure
#  - stdout, stderr
#  - other module instance info?
#  - qmmm minimisation config:
#    minimisation = Some (exess_rex::ClassicalMinimisation {
#      err_tol_kj_per_mol_nm = $err_tol_kj_per_mol_nm,
#      max_iterations = $max_iterations,
#    }),
