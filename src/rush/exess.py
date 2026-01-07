#!/usr/bin/env python3
"""
EXESS module helpers for the Rush Python client.

EXESS supports whole-system energy calculations (fragmented or unfragmented),
interaction energy between a fragment and the rest of the system, geometry
optimization, simulations, and gradient/Hessian calculations. It supports
multiple levels of theory (e.g., restricted/unrestricted HF, RI-MP2, DFT),
flexible basis set selection, and configurable n-mer fragmentation levels.

Quick Links
-----------

- :func:`rush.exess.exess`
- :func:`rush.exess.energy`
- :func:`rush.exess.interaction_energy`
- :func:`rush.exess.chelpg`
- :func:`rush.exess.qmmm`
- :func:`rush.exess.optimization`


"""

import sys
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Literal

import h5py
import zstandard as zstd
from gql.transport.exceptions import TransportQueryError

from .client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    _submit_rex,
    collect_run,
    download_object,
    save_json,
    save_object,
    upload_object,
)
from .utils import bool_to_str, float_to_str, optional_str

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

type StandardOrientationT = Literal[
    "None",
    "FullSystem",
    "PerFragment",
]


@dataclass
class Model:
    #: Determines if the system is tranformed into a "standard orientation"
    #: during the calculations. (Default: "FullSystem") Setting this value to "None"
    #: prevents any transformation from happening, such that the output is exactly
    #: aligned with the input.
    standard_orientation: StandardOrientationT | None = None

    #: Determines whether spherical or Cartesian basis sets will be used.
    #: (Default: "True") Setting this value to "False" could provide speedup or memory
    #: savings in some cases, but certain features require Cartesian basis sets.
    force_cartesian_basis_sets: bool | None = None

    def _to_rex(self, method: MethodT, basis: BasisT, aux_basis: AuxBasisT):
        return Template(
            """Some (exess_rex::Model {
          method = exess_rex::Method::$method,
          basis = "$basis",
          aux_basis = $maybe_aux_basis,
          standard_orientation = $maybe_standard_orientation,
          force_cartesian_basis_sets = $maybe_force_cartesian_basis_sets,
        })"""
        ).substitute(
            method=method,
            basis=basis,
            maybe_aux_basis=optional_str(aux_basis),
            maybe_standard_orientation=optional_str(
                self.standard_orientation, "exess_rex::StandardOrientation::"
            ),
            maybe_force_cartesian_basis_sets=optional_str(
                self.force_cartesian_basis_sets
            ),
        )


@dataclass
class System:
    #: Maximum memory to allocate to the GPU for EXESS's dedicated use.
    #: Try setting this to limit or increase the memory if EXESS's automatic
    #: determination of how much to allocate is not working properly
    #: (and probably file a bug too).
    max_gpu_memory_mb: int | None = None

    #: Allow EXESS to over-allocate memory on GPUs.
    oversubscribe_gpus: bool | None = None

    #: Sets corresponding MPI configuration.
    gpus_per_team: int | None = None

    #: Sets corresponding MPI configuration.
    teams_per_node: int | None = None

    def _to_rex(self):
        return Template(
            """Some (exess_rex::System {
          max_gpu_memory_mb = $maybe_max_gpu_memory_mb,
          oversubscribe_gpus = $maybe_oversubscribe_gpus,
          gpus_per_team = $maybe_gpus_per_team,
          teams_per_node = $maybe_teams_per_node,
        })"""
        ).substitute(
            maybe_max_gpu_memory_mb=optional_str(self.max_gpu_memory_mb),
            maybe_oversubscribe_gpus=optional_str(self.oversubscribe_gpus),
            maybe_gpus_per_team=optional_str(self.gpus_per_team),
            maybe_teams_per_node=optional_str(self.teams_per_node),
        )


type ConvergenceMetricT = Literal[
    "Energy",
    "DIIS",
    "Density",
]

type FockBuildTypeT = Literal[
    "HGP",
    "UM09",
    "RI",
]


@dataclass
class SCFKeywords:
    #: Max SCF iterations performed. Ajust depending on the convergence_threshold chosen.
    max_iters: int = 50
    #: Use this keyword to control the size of the DIIS extrapolation space, i.e.
    #: how many past iteration matrices will be used to extrapolate the Fock matrix.
    #: A larger number will result in slightly higher memory use.
    #: This can become a problem when dealing with large systems without fragmentation.
    max_diis_history_length: int = 8
    #: Number of shell pair batches stored in the shell-pair batch bin container.
    batch_size: int = 2560
    #: Metric to use for SCF convergence. Using energy as the convergence metric can
    #: lead to early convergence which can produce unideal orbitals for MP2 calculations.
    convergence_metric: ConvergenceMetricT = "DIIS"
    #: SCF convergence threshold
    convergence_threshold: float = 1e-6
    #: Besides the Cauchy-Schwarz screening, inside each integral kernel
    #: the integrals are further screened against the density matrix.
    #: This threshold controls at which value an integral is considered to be negligible.
    #: Decreasing this threshold will lead to significantly faster SCF times
    #: at the possible cost of accuracy.
    #: Increasing it to 1E-11 and 1E-12 will lead to longer SCF times because
    #: more integrals will be evaluated. However, for methods such as tetramer level MBE
    #: this can better the accuracy of the program.
    #: This will also produce crisper orbitals for MP2 calculations.
    density_threshold: float = 1e-10
    #: Like the density, the integrals are further screened against the gradient matrix.
    gradient_screening_threshold: float = 1e-10
    bf_cutoff_threshold: float | None = None
    #: Fall back to STO-3G basis set for calcuulation and project up
    #: if SCF is unconverged (Default: True)
    density_basis_set_projection_fallback_enabled: bool | None = None
    use_ri: bool = False
    store_ri_b_on_host: bool = False
    #: Compress the B matrix for RI-HF (Default: False)
    compress_ri_b: bool = False
    homo_lumo_guess_rotation_angle: float | None = None
    # Select type of fock build algorithm, Options: [“HGP”, “UM09”, “RI”]
    fock_build_type: FockBuildTypeT = "HGP"
    exchange_screening_threshold: float = 1e-5
    group_shared_exponents: bool = False

    def _to_rex(self):
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


type FragmentLevelT = Literal[
    "Monomer",
    "Dimer",
    "Trimer",
    "Tetramer",
]
type CutoffTypeT = Literal["Centroid", "ClosestPair"]
type DistanceMetricT = Literal["Max", "Average", "Min"]


@dataclass
class FragKeywords:
    """
    Configure the fragmentation of the system.

    Defaults are provided for all relevant levels.
    NOTE: cutoffs for each level must be less than or equal to those at the lower levels.
    """

    #: Controls at which level the many body expansion is truncated.
    #: I.e., what order of n-mers to create fragments for when fragmenting.
    #: Reasonable values range from Dimer to Tetramer, with Dimers being a quick and
    #: efficient but still meaningful initial configuration when experimenting.
    level: FragmentLevelT = "Dimer"
    #: The cutoffs control at what distance a polymer won’t be calculated.
    #: All distances are in Angstroms.
    dimer_cutoff: float | None = None
    #: See documentation for dimer_cutoff.
    trimer_cutoff: float | None = None
    #: See documentation for dimer_cutoff.
    tetramer_cutoff: float | None = None
    #: Default is "ClosestPair", which uses the closest pair of atoms in each fragment
    #: to assess their distance rather than the distance between fragment centroids.
    cutoff_type: CutoffTypeT | None = None
    distance_metric: DistanceMetricT | None = None
    #: Calculation will act as if only those fragments were present.
    included_fragments: list[int] | None = None
    enable_speed: bool | None = None

    def __post_init__(self):
        if self.level == "Monomer":
            self.dimer_cutoff = 100.0
            self.trimer_cutoff = None
            self.tetramer_cutoff = None
            self.cutoff_type = None
            self.distance_metric = None
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

    def _to_rex(self, reference_fragment: int | None = None):
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
            distance_metric = $maybe_distance_metric,
            level = exess_rex::FragmentLevel::$level,
            included_fragments = $maybe_included_fragments,
            reference_fragment = $maybe_reference_fragment,
            enable_speed = $maybe_enable_speed,
          })"""
        ).substitute(
            dimer_cutoff=optional_str(self.dimer_cutoff),
            trimer_cutoff=optional_str(self.trimer_cutoff),
            tetramer_cutoff=optional_str(self.tetramer_cutoff),
            maybe_cutoff_type=optional_str(
                self.cutoff_type, "exess_rex::FragmentDistanceMethod::"
            ),
            maybe_distance_metric=optional_str(
                self.distance_metric, "exess_rex::FragmentDistanceMetric::"
            ),
            level=self.level,
            maybe_included_fragments=optional_str(self.included_fragments),
            maybe_reference_fragment=optional_str(reference_fragment),
            maybe_enable_speed=optional_str(self.enable_speed),
        )


@dataclass
class StandardDescriptorGrid:
    """
    Constructs a "standard" descriptor grid.
    """

    value: Literal["SG1", "SG2"]

    def _to_rex(self):
        return Template(
            """Some (
              exess_rex::DescriptorGridOptions::standard exess_rex::StandardGrid::$value
            )""",
        ).substitute(
            value=self.value,
        )


@dataclass
class DescriptorGrid:
    """
    Constructs a descriptor grid based on the parameters.
    """

    points_per_shell: int
    order: Literal["One", "Two"]
    scale: float

    def _to_rex(self):
        return Template(
            """Some (exess_rex::DescriptorGridOptions::params (
              exess_rex::Grid {
                points_per_shell = $points_per_shell,
                order = exess_rex::GridOrder::$order,
                scale = $scale,
              }
            ))"""
        ).substitute(
            points_per_shell=self.points_per_shell,
            order=self.order,
            scale=float_to_str(self.scale),
        )


@dataclass
class CustomDescriptorGrid:
    """
    Construct a totally custom descriptor grid with each point being explicitly
    specified by its (x, y, z) coordinates. Points are specified one after the other,
    e.g. [x1, y1, z1, x2, y2, z2, ...].
    """

    value: list[float]

    def _to_rex(self):
        return Template(
            """Some (
              exess_rex::DescriptorGridOptions::custom $value
            )"""
        ).substitute(
            value=f"[{', '.join([float_to_str(float(v)) for v in self.value])}]",
        )


@dataclass
class ExportKeywords:
    """
    Configure the exported outputs of the system.
    Outputs are in both JSON and HDF5 format (some just one or the other).
    Most outputs are in the HDF5 file only.
    """

    #: Electron density
    export_density: bool | None = None
    #: Relaxed MP2 density correction (?)
    export_relaxed_mp2_density_correction: bool | None = None
    #: Fock matrix (?)
    export_fock: bool | None = None
    #: Overlap matrix (?)
    export_overlap: bool | None = None
    #: H core matrix
    export_h_core: bool | None = None
    #: Provides the whole density matrix for entire fragment system,
    #: rather than per-fragment matrices.
    export_expanded_density: bool | None = None
    #: Provides the whole gradient matrix for entire fragment system,
    #: rather than per-fragment matrices.
    #: NOTE: If set, must be performing a gradient calculation.
    export_expanded_gradient: bool | None = None
    #: Fancy... (?)
    export_molecular_orbital_coeffs: bool | None = None
    #: Energy gradient values (as used in Optimization and QMMM).
    #: NOTE: If set, must be performing a gradient calculation.
    export_gradient: bool | None = None
    #: If external charges are used, export the gradient for these point charges.
    export_external_charge_gradient: bool | None = None
    #: Mulliken charges for the atoms in the system.
    export_mulliken_charges: bool | None = None
    #: ChelpG partial charges for the atoms in the system.
    export_chelpg_charges: bool | None = None
    #: Believed to be a pass-through from the input connectivity.
    export_bond_orders: bool | None = None
    #: The generated hydrogen caps for fragments in fragmented systems.
    export_h_caps: bool | None = None
    #: Derived values from electron density.
    export_density_descriptors: bool | None = None
    #: Derived values from electrostatic potential.
    export_esp_descriptors: bool | None = None
    #: Provides the whole esp descriptor matrix for entire fragment system,
    #: rather than per-fragment matrices. NOTE: Causes memory errors.
    export_expanded_esp_descriptors: bool | None = None
    # Provides the basis sets used (?).
    export_basis_labels: bool | None = None
    # Hessian tensor.
    #: NOTE: If set, must be performing a Hessian calculation.
    export_hessian: bool | None = None
    # ?
    export_mass_weighted_hessian: bool | None = None
    # ?
    export_hessian_frequencies: bool | None = None
    # When exporting square symmetric matrices, save memory by exporting the flattened
    #: lower triangle of the matrix. (Default: True)
    flatten_symmetric: bool | None = None
    # ?
    light_json: bool | None = None
    # Post-process exports into a single HDF5 output file.
    # This is relevant for fragmented runs (particularly when configured for multinode).
    # The concatenation of the HDF5 files may be expensive.
    concatenate_hdf5_files: bool | None = None
    # ?
    training_db: bool | None = None
    # Grid of points at which to calculate and export density descriptors.
    descriptor_grid: (
        StandardDescriptorGrid | DescriptorGrid | CustomDescriptorGrid | None
    ) = None

    def _to_rex(self):
        return Template(
            """Some (exess_rex::ExportKeywords {
            export_density = $maybe_export_density,
            export_relaxed_mp2_density_correction = $maybe_export_relaxed_mp2_density_correction,
            export_fock = $maybe_export_fock,
            export_overlap = $maybe_export_overlap,
            export_h_core = $maybe_export_h_core,
            export_expanded_density = $maybe_export_expanded_density,
            export_expanded_gradient = $maybe_export_expanded_gradient,
            export_molecular_orbital_coeffs = $maybe_export_molecular_orbital_coeffs,
            export_gradient = $maybe_export_gradient,
            export_external_charge_gradient = $maybe_export_external_charge_gradient,
            export_mulliken_charges = $maybe_export_mulliken_charges,
            export_chelpg_charges = $maybe_export_chelpg_charges,
            export_bond_orders = $maybe_export_bond_orders,
            export_h_caps = $maybe_export_h_caps,
            export_density_descriptors = $maybe_export_density_descriptors,
            export_esp_descriptors = $maybe_export_esp_descriptors,
            export_expanded_esp_descriptors = $maybe_export_expanded_esp_descriptors,
            export_basis_labels = $maybe_export_basis_labels,
            export_hessian = $maybe_export_hessian,
            export_mass_weighted_hessian = $maybe_export_mass_weighted_hessian,
            export_hessian_frequencies = $maybe_export_hessian_frequencies,
            flatten_symmetric = $maybe_flatten_symmetric,
            light_json = $maybe_light_json,
            concatenate_hdf5_files = $maybe_concatenate_hdf5_files,
            training_db = $maybe_training_db,
            descriptor_grid = $maybe_descriptor_grid,
          })"""
        ).substitute(
            maybe_export_density=optional_str(self.export_density),
            maybe_export_relaxed_mp2_density_correction=optional_str(
                self.export_relaxed_mp2_density_correction
            ),
            maybe_export_fock=optional_str(self.export_fock),
            maybe_export_overlap=optional_str(self.export_overlap),
            maybe_export_h_core=optional_str(self.export_h_core),
            maybe_export_expanded_density=optional_str(self.export_expanded_density),
            maybe_export_expanded_gradient=optional_str(self.export_expanded_gradient),
            maybe_export_molecular_orbital_coeffs=optional_str(
                self.export_molecular_orbital_coeffs
            ),
            maybe_export_gradient=optional_str(self.export_gradient),
            maybe_export_external_charge_gradient=optional_str(
                self.export_external_charge_gradient
            ),
            maybe_export_mulliken_charges=optional_str(self.export_mulliken_charges),
            maybe_export_chelpg_charges=optional_str(self.export_chelpg_charges),
            maybe_export_bond_orders=optional_str(self.export_bond_orders),
            maybe_export_h_caps=optional_str(self.export_h_caps),
            maybe_export_density_descriptors=optional_str(
                self.export_density_descriptors
            ),
            maybe_export_esp_descriptors=optional_str(self.export_esp_descriptors),
            maybe_export_expanded_esp_descriptors=optional_str(
                self.export_expanded_esp_descriptors
            ),
            maybe_export_basis_labels=optional_str(self.export_basis_labels),
            maybe_export_hessian=optional_str(self.export_hessian),
            maybe_export_mass_weighted_hessian=optional_str(
                self.export_mass_weighted_hessian
            ),
            maybe_export_hessian_frequencies=optional_str(
                self.export_hessian_frequencies
            ),
            maybe_flatten_symmetric=optional_str(self.flatten_symmetric),
            maybe_light_json=optional_str(self.light_json),
            maybe_concatenate_hdf5_files=optional_str(self.concatenate_hdf5_files),
            maybe_training_db=optional_str(self.training_db),
            maybe_descriptor_grid=(
                self.descriptor_grid._to_rex()
                if self.descriptor_grid is not None
                else "None"
            ),
        )


@dataclass
class Trajectory:
    """
    Configure the output of QMMM runs. By default, will provide all atoms at every frame.
    """

    #: Save every n frames to the trajectory, where n is the interval specified.
    interval: int | None = None
    #: The frame at which to start the trajectory.
    start: int | None = None
    #: The frame at which to end the trajectory.
    end: int | None = None
    #: Whether to include waters in the trajectory. Convenient for reducing output size.
    include_waters: int | None = None

    def _to_rex(self):
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

    #: Scaling factor for restraints (larger values mean a stronger restraint).
    k: float | None = None
    #: Which atoms to hold fixed. All fixed/free parameters are mutually exclusive.
    fixed_atoms: list[int] | None = None
    #: Which atoms to keep unfixed. All fixed/free parameters are mutually exclusive.
    free_atoms: list[int] | None = None
    #: Which fragments to hold fixed. All fixed/free parameters are mutually exclusive.
    fixed_fragments: list[int] | None = None
    #: Which fragments to keep unfixed. All fixed/free parameters are mutually exclusive.
    free_fragments: list[int] | None = None
    #: Flag to easily enable fixing all heavy atoms only. Mutually exclusive with fixed/free parameters.
    fix_heavy: bool | None = None

    def _to_rex(self):
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


def exess(
    topology_path: Path | str,
    driver: str = "Energy",
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords | None = FragKeywords(),
    export_keywords: ExportKeywords | None = ExportKeywords(),
    system: System | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the energy of the system in the QDX topology file at `topology_path`.
    """

    # Upload inputs
    topology_vobj = upload_object(topology_path)

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
          standard_orientation = $maybe_standard_orientation,
          force_cartesian_basis_sets = $maybe_force_cartesian_basis_sets,
        }),
        system = $maybe_system,
        keywords = exess_rex::Keywords {
          scf = $maybe_scf_keywords,
          ks = None,
          rtat = None,
          frag = $maybe_frag_keywords,
          boundary = None,
          log = None,
          dynamics = None,
          integrals = None,
          debug = None,
          export = $maybe_export_keywords,
          guess = None,
          force_field = None,
          optimization = None,
          hessian = None,
          gradient = None,
          qmmm = None,
          machine_learning = None,
          regions = None,
        },
        driver = exess_rex::Driver::$driver,
      })
      [ (obj_j topology) ]
      None
in
  exess "$topology_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(aux_basis),
        maybe_standard_orientation=optional_str(
            standard_orientation, "exess_rex::StandardOrientation::"
        ),
        maybe_force_cartesian_basis_sets=optional_str(force_cartesian_basis_sets),
        maybe_system=system._to_rex() if system is not None else "None",
        maybe_scf_keywords=(
            scf_keywords._to_rex() if scf_keywords is not None else "None"
        ),
        maybe_frag_keywords=(
            frag_keywords._to_rex() if frag_keywords is not None else "None"
        ),
        maybe_export_keywords=(
            export_keywords._to_rex() if export_keywords is not None else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        driver=driver,
    )
    try:
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def energy(
    topology_path: Path | str,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords | None = FragKeywords(),
    export_keywords: ExportKeywords | None = ExportKeywords(),
    system: System | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    return exess(
        topology_path,
        "Energy",
        method,
        basis,
        aux_basis,
        standard_orientation,
        force_cartesian_basis_sets,
        scf_keywords,
        frag_keywords,
        export_keywords,
        system,
        run_spec,
        run_opts,
        collect,
    )


def save_energy_outputs(res, to_json=False):
    if len(res) == 1:
        return save_object(res[0]["path"])
    else:
        qm_output = download_object(res[1]["path"])
        decompressed = zstd.ZstdDecompressor().decompress(
            qm_output, max_output_size=int(1e9)
        )
        with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
            hdf5_f = tar.extractfile(tar.getnames()[1])
            with h5py.File(hdf5_f, "r") as f:
                if "monomers" not in f.keys():
                    # Our outer key will be the exported val
                    exported_vals = [x for x in f.keys()]
                    d = {k: {} for k in exported_vals}
                    for exported_val in exported_vals:
                        d[exported_val] = f[exported_val][()].tolist()
                else:
                    # Check if we have any fragments (we probably should)
                    frag_indices = [int(x) for x in f["monomers"].keys()]
                    if not frag_indices:
                        return save_object(res[0]["path"])

                    # Check if anything got exported
                    exported_vals = [x for x in f[f"monomers/{frag_indices[0]}"].keys()]
                    if not exported_vals:
                        return save_object(res[0]["path"])

                    # Our outer key will be the exported val
                    d = {k: {} for k in exported_vals}
                    for exported_val in exported_vals:
                        for nmer_type in f.keys():
                            if nmer_type not in d[exported_val]:
                                d[exported_val][nmer_type] = {}
                            for nmer_idx in sorted(f[f"{nmer_type}"].keys()):
                                d[exported_val][nmer_type][nmer_idx] = f[
                                    f"{nmer_type}/{nmer_idx}/{exported_val}"
                                ][()].tolist()

        return (
            save_object(res[0]["path"]),
            (
                save_json(d, name=res[1]["path"])
                if to_json
                else save_object(res[1]["path"], ext="hdf5", extract=True)
            ),
        )


def interaction_energy(
    topology_path: Path | str,
    reference_fragment: int,
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords = FragKeywords(),
    system: System | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the interaction energy between the fragment with index `reference_fragment` and the rest of the system
    in the toplogy file at `topology_path`.
    """

    # Upload inputs
    topology_vobj = upload_object(topology_path)

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
          standard_orientation = $maybe_standard_orientation,
          force_cartesian_basis_sets = $maybe_force_cartesian_basis_sets,
        }),
        system = $maybe_system,
        keywords = exess_rex::Keywords {
          scf = $maybe_scf_keywords,
          ks = None,
          rtat = None,
          frag = $maybe_frag_keywords,
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
        run_spec=run_spec._to_rex(),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(aux_basis),
        maybe_standard_orientation=optional_str(
            standard_orientation, "exess_rex::StandardOrientation::"
        ),
        maybe_force_cartesian_basis_sets=optional_str(force_cartesian_basis_sets),
        maybe_system=system._to_rex() if system is not None else "None",
        maybe_scf_keywords=(
            scf_keywords._to_rex() if scf_keywords is not None else "None"
        ),
        maybe_frag_keywords=frag_keywords._to_rex(reference_fragment),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def chelpg(
    topology_path: Path | str,
    system: System | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the CHELPG partial charges for all atoms of the system in the topology file at `topology_path`.
    """

    # Upload inputs
    topology_vobj = upload_object(topology_path)

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
          standard_orientation = Some exess_rex::StandardOrientation::None,
          force_cartesian_basis_sets = Some false,
        }),
        system = $system,
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
        run_spec=run_spec._to_rex(),
        system=system._to_rex() if system is not None else "None",
        scf_keywords=SCFKeywords(
            max_diis_history_length=12, convergence_threshold=1e-8
        )._to_rex(),
        frag_keywords=FragKeywords(level="Monomer")._to_rex(),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            result = collect_run(run_id)
            qm_output = download_object(result[1]["path"])
            decompressed = zstd.ZstdDecompressor().decompress(
                qm_output, max_output_size=int(1e9)
            )
            with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
                hdf5_f = tar.extractfile(tar.getnames()[1])
                with h5py.File(hdf5_f, "r") as f:
                    frag_indices = [int(x) for x in f["monomers"].keys()]
                    chelpg = [
                        float(x)
                        for frag_idx in sorted(frag_indices)
                        for x in f[f"monomers/{frag_idx}/chelpg_charges"]
                    ]
            return [result[0], chelpg]
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


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
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords = FragKeywords(),
    qm_fragments: list[int] | None = None,
    mm_fragments: list[int] | None = None,
    ml_fragments: list[int] | None = None,
    system: System | None = None,
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
    topology_vobj = upload_object(topology_path)
    residues_vobj = upload_object(residues_path)

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
          standard_orientation = $maybe_standard_orientation,
          force_cartesian_basis_sets = $maybe_force_cartesian_basis_sets,
        }),
        system = $system,
        keywords = exess_qmmm_rex::Keywords {
          scf = $maybe_scf_keywords,
          ks = None,
          rtat = None,
          frag = $maybe_frag_keywords,
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
            restraints = $maybe_restraints,
            energy_csv = None,
          }),
          machine_learning = $maybe_machine_learning,
          regions = $maybe_regions,
        },
      })
      (obj_j topology)
      (Some (obj_j residues))
in
  exess "$topology_vobj_path" "$residues_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(aux_basis),
        maybe_standard_orientation=optional_str(
            standard_orientation, "exess_rex::StandardOrientation::"
        ),
        maybe_force_cartesian_basis_sets=optional_str(force_cartesian_basis_sets),
        system=system._to_rex() if system is not None else "None",
        maybe_scf_keywords=(
            scf_keywords._to_rex() if scf_keywords is not None else "None"
        ),
        maybe_frag_keywords=(
            frag_keywords._to_rex() if frag_keywords is not None else "None"
        ),
        maybe_gradient_finite_difference_step_size=optional_str(
            gradient_finite_difference_step_size
        ),
        n_timesteps=n_timesteps,
        dt_ps=dt_ps,
        temperature_kelvin=temperature_kelvin,
        maybe_pressure_atm=optional_str(pressure_atm),
        trajectory=trajectory._to_rex(),
        maybe_restraints=restraints._to_rex() if restraints is not None else "None",
        maybe_machine_learning=(
            "Some (exess_geo_opt_rex::MLKeywords { ml_type = None })"
            if ml_fragments is not None
            else "None"
        ),
        maybe_regions=(
            Template(
                """Some (exess_qmmm_rex::RegionKeywords {
            qm_fragments = $maybe_qm_fragments,
            mm_fragments = $maybe_mm_fragments,
            ml_fragments = $maybe_ml_fragments,
          })"""
            ).substitute(
                maybe_qm_fragments=optional_str(qm_fragments),
                maybe_mm_fragments=optional_str(mm_fragments),
                maybe_ml_fragments=optional_str(ml_fragments),
            )
            if not (
                qm_fragments is None and mm_fragments is None and ml_fragments is None
            )
            else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"],
    )
    try:
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


@dataclass
class OptimizationConvergenceCriteria:
    metric: str | None = None
    gradient_threshold: float | None = None
    delta_energy_threshold: float | None = None
    step_component_threshold: float | None = None

    def _to_rex(self, reference_fragment: int | None = None):
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

    def _to_rex(self):
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
    linesearch: LBFGSLinesearchT = "BacktrackingStrongWolfe"
    n_corrections: int | None = None
    epsilon: float | None = None
    max_linesearch: int | None = None
    gtol: float | None = None

    def _to_rex(self):
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

    def _to_rex(self, max_iters):
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
                self.convergence_criteria._to_rex()
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
                self.lbfgs_keywords._to_rex()
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
                self.trust_region_keywords._to_rex()
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
    residues_path: Path | str | None = None,
    optimization_keywords: OptimizationKeywords = OptimizationKeywords(),
    method: MethodT = "RestrictedHF",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    qm_fragments: list[int] | None = None,
    mm_fragments: list[int] | None = None,
    ml_fragments: list[int] | None = None,
    system: System | None = None,
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
    topology_vobj = upload_object(topology_path)
    residues_vobj = None
    if residues_path is not None:
        residues_vobj = upload_object(residues_path)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology residues →
    exess_geo_opt_rex_s
      ($run_spec)
      (exess_geo_opt_rex::OptimizationParams {
        schema_version = "0.2.0",
        external_charges = None,
        model = Some (exess_geo_opt_rex::Model {
          method = exess_geo_opt_rex::Method::$method,
          basis = "$basis",
          aux_basis = $maybe_aux_basis,
          standard_orientation = $maybe_standard_orientation,
          force_cartesian_basis_sets = $maybe_force_cartesian_basis_sets,
        }),
        system = $maybe_system,
        keywords = exess_geo_opt_rex::Keywords {
          scf = $maybe_scf_keywords,
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
          optimization = $maybe_optimization_keywords,
          hessian = None,
          gradient = None,
          qmmm = None,
          machine_learning = $maybe_machine_learning,
          regions = $maybe_regions,
        },
      })
      [ (obj_j topology) ]
      $residues_expr
in
  exess "$topology_vobj_path" "$residues_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        method=method,
        basis=basis,
        maybe_aux_basis=optional_str(aux_basis),
        maybe_standard_orientation=optional_str(
            standard_orientation, "exess_rex::StandardOrientation::"
        ),
        maybe_force_cartesian_basis_sets=optional_str(force_cartesian_basis_sets),
        maybe_system=system._to_rex() if system is not None else "None",
        maybe_scf_keywords=(
            scf_keywords._to_rex() if scf_keywords is not None else "None"
        ),
        maybe_optimization_keywords=(
            optimization_keywords._to_rex(max_iters)
            if optimization_keywords is not None
            else "None"
        ),
        maybe_machine_learning=(
            "Some (exess_geo_opt_rex::MLKeywords { ml_type = None })"
            if ml_fragments is not None
            else "None"
        ),
        maybe_regions=(
            Template(
                """Some (exess_qmmm_rex::RegionKeywords {
            qm_fragments = $maybe_qm_fragments,
            mm_fragments = $maybe_mm_fragments,
            ml_fragments = $maybe_ml_fragments,
          })"""
            ).substitute(
                maybe_qm_fragments=optional_str(qm_fragments),
                maybe_mm_fragments=optional_str(mm_fragments),
                maybe_ml_fragments=optional_str(ml_fragments),
            )
            if not (
                qm_fragments is None and mm_fragments is None and ml_fragments is None
            )
            else "None"
        ),
        residues_expr=(
            "(Some [ (obj_j residues) ])" if residues_path is not None else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"] if residues_vobj is not None else "",
    )
    try:
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


# TODO:
#  - trace for failure
#  - stdout, stderr
#  - other module instance info?
#  - qmmm minimisation config:
#    minimisation = Some (exess_rex::ClassicalMinimisation {
#      err_tol_kj_per_mol_nm = $err_tol_kj_per_mol_nm,
#      max_iterations = $max_iterations,
#    }),
