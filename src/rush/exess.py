#!/usr/bin/env python3
"""
EXESS module helpers for the Rush Python client.

EXESS supports whole-system energy calculations (fragmented or unfragmented),
interaction energy between a fragment and the rest of the system, and
gradient/Hessian calculations. It supports multiple levels of theory
(e.g., restricted/unrestricted HF, RI-MP2, DFT), flexible basis set
selection, and configurable n-mer fragmentation levels.

Quick Links
-----------

- :func:`rush.exess.exess`
- :func:`rush.exess.exess_energy`
- :func:`rush.exess.exess_interaction_energy`
- :func:`rush.exess.fetch_outputs`
- :func:`rush.exess.save_outputs`
- :mod:`rush.exess_geo_opt`
- :mod:`rush.exess_qmmm`
"""

import enum
import json
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from string import Template
from typing import Any, Literal

from gql.transport.exceptions import TransportQueryError

from .client import (
    RunError,
    RunOpts,
    RunSpec,
    _get_project_id,
    _submit_rex,
    collect_run,
    fetch_object,
    save_object,
    upload_object,
)
from .mol import AtomRef, FragmentRef
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
    "def2-SVP",
    "def2-TZVP",
    "def2-TZVPD",
    "def2-TZVPP",
    "def2-TZVPPD",
]

type AuxBasisT = Literal[
    "6-31G**-RIFIT",
    "aug-cc-pVDZ-RIFIT",
    "aug-cc-pVTZ-RIFIT",
    "cc-pVDZ-RIFIT",
    "cc-pVTZ-RIFIT",
    "def2-SVP-RIFIT",
    "def2-TZVP-RIFIT",
    "def2-TZVPD-RIFIT",
    "def2-TZVPP-RIFIT",
    "def2-TZVPPD-RIFIT",
]

type StandardOrientationT = Literal[
    "None",
    "FullSystem",
    "PerFragment",
]


type TensorLike = list[Any]


@dataclass
class ExessNmer:
    schema_version: str
    fragments: list[FragmentRef]
    density: TensorLike | None = None
    fock: TensorLike | None = None
    overlap: TensorLike | None = None
    h_core: TensorLike | None = None
    coeffs_initial: TensorLike | None = None
    coeffs_final: TensorLike | None = None
    molecular_orbital_energies: list[float] | None = None
    hf_gradients: list[float] | None = None
    mp2_gradients: list[float] | None = None
    hf_energy: float | None = None
    mp2_ss_correction: float | None = None
    mp2_os_correction: float | None = None
    ccsd_correction: float | None = None
    s_squared_eigenvalue: float | None = None
    delta_hf_energy: float | None = None
    delta_mp2_ss_correction: float | None = None
    delta_mp2_os_correction: float | None = None
    mulliken_charges: list[float] | None = None
    chelpg_charges: list[float] | None = None
    fragment_distance: float | None = None
    bond_orders: list[list[float]] | None = None
    h_caps: list[AtomRef] | None = None
    num_iters: int | None = None
    num_basis_fns: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExessNmer":
        return cls(
            schema_version=data["schema_version"],
            fragments=[FragmentRef(fragment) for fragment in data["fragments"]],
            density=data.get("density"),
            fock=data.get("fock"),
            overlap=data.get("overlap"),
            h_core=data.get("h_core"),
            coeffs_initial=data.get("coeffs_initial"),
            coeffs_final=data.get("coeffs_final"),
            molecular_orbital_energies=data.get("molecular_orbital_energies"),
            hf_gradients=data.get("hf_gradients"),
            mp2_gradients=data.get("mp2_gradients"),
            hf_energy=data.get("hf_energy"),
            mp2_ss_correction=data.get("mp2_ss_correction"),
            mp2_os_correction=data.get("mp2_os_correction"),
            ccsd_correction=data.get("ccsd_correction"),
            s_squared_eigenvalue=data.get("s_squared_eigenvalue"),
            delta_hf_energy=data.get("delta_hf_energy"),
            delta_mp2_ss_correction=data.get("delta_mp2_ss_correction"),
            delta_mp2_os_correction=data.get("delta_mp2_os_correction"),
            mulliken_charges=data.get("mulliken_charges"),
            chelpg_charges=data.get("chelpg_charges"),
            fragment_distance=data.get("fragment_distance"),
            bond_orders=data.get("bond_orders"),
            h_caps=(
                [AtomRef(atom) for atom in data["h_caps"]]
                if data.get("h_caps") is not None
                else None
            ),
            num_iters=data.get("num_iters"),
            num_basis_fns=data.get("num_basis_fns"),
        )


@dataclass
class ExessManyBodyExpansion:
    schema_version: str
    method: str
    nmers: list[list[ExessNmer]]
    distance_metric: str | None = None
    distance_method: str | None = None
    reference_fragment: FragmentRef | None = None
    expanded_hf_energy: float | None = None
    classical_water_energy: float | None = None
    expanded_mp2_ss_correction: float | None = None
    expanded_mp2_os_correction: float | None = None
    expanded_ccsd_correction: float | None = None
    expanded_density: TensorLike | None = None
    expanded_hf_gradients: list[float] | None = None
    expanded_mp2_gradients: list[float] | None = None
    num_iters: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExessManyBodyExpansion":
        return cls(
            schema_version=data["schema_version"],
            method=data["method"],
            nmers=[
                [ExessNmer.from_dict(nmer) for nmer in nmer_level]
                for nmer_level in data["nmers"]
            ],
            distance_metric=data.get("distance_metric"),
            distance_method=data.get("distance_method"),
            reference_fragment=(
                FragmentRef(data["reference_fragment"])
                if data.get("reference_fragment") is not None
                else None
            ),
            expanded_hf_energy=data.get("expanded_hf_energy"),
            classical_water_energy=data.get("classical_water_energy"),
            expanded_mp2_ss_correction=data.get("expanded_mp2_ss_correction"),
            expanded_mp2_os_correction=data.get("expanded_mp2_os_correction"),
            expanded_ccsd_correction=data.get("expanded_ccsd_correction"),
            expanded_density=data.get("expanded_density"),
            expanded_hf_gradients=data.get("expanded_hf_gradients"),
            expanded_mp2_gradients=data.get("expanded_mp2_gradients"),
            num_iters=data.get("num_iters"),
        )


@dataclass
class ExessCalculation:
    schema_version: str
    calculation_time: float
    qmmbe: ExessManyBodyExpansion | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExessCalculation":
        return cls(
            schema_version=data["schema_version"],
            calculation_time=data["calculation_time"],
            qmmbe=(
                ExessManyBodyExpansion.from_dict(data["qmmbe"])
                if data.get("qmmbe") is not None
                else None
            ),
        )


@dataclass
class ExessResult:
    calc: ExessCalculation
    exports: dict[str, Any] | bytes | None = None


@dataclass(frozen=True)
class ExessSavedResult:
    calc: Path
    exports: Path | None = None


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
    included_fragments: list[int | FragmentRef] | None = None
    #: Enables interaction-energy mode for the selected fragment.
    reference_fragment: int | None = None
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

    def _to_rex(self):
        included_fragments = None
        if self.included_fragments:
            included_fragments = [
                f.value if isinstance(f, FragmentRef) else f
                for f in self.included_fragments
            ]
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
            maybe_included_fragments=optional_str(included_fragments),
            maybe_reference_fragment=optional_str(self.reference_fragment),
            maybe_enable_speed=optional_str(self.enable_speed),
        )


@dataclass
class StandardDescriptorGrid:
    """
    Constructs a "standard" descriptor grid.
    """

    value: Literal[
        "Fine",  #: Default
        "UltraFine",
        "SuperFine",
        "TreutlerGM3",
        "TreutlerGM5",
    ]

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
class RegularDescriptorGrid:
    """
    Construct a regular Cartesian descriptor grid with evenly-spaced points between
    the minimum and maximum points specified, at the defined spacing in each dimension.
    """

    min: list[float]
    max: list[float]
    spacing: list[float]

    def _to_rex(self):
        return Template(
            """Some (exess_rex::DescriptorGridOptions::regular (
              exess_rex::RegularGrid {
                min = $min,
                max = $max,
                spacing = $spacing,
              }
            ))"""
        ).substitute(
            min=f"[{', '.join([float_to_str(float(v)) for v in self.min])}]",
            max=f"[{', '.join([float_to_str(float(v)) for v in self.max])}]",
            spacing=f"[{', '.join([float_to_str(float(v)) for v in self.spacing])}]",
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
        StandardDescriptorGrid
        | DescriptorGrid
        | CustomDescriptorGrid
        | RegularDescriptorGrid
        | None
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


type RadialQuadT = Literal[
    "MuraKnowles",  #: Default
    "MurrayHandyLaming",
    "TreutlerAldrichs",
]

type PruningSchemeT = Literal[
    "Unpruned",
    "Robust",  #: Default
    "Treutler",
]


@dataclass
class DefaultGridResolution:
    default_grid: Literal[
        "Fine",
        "UltraFine",  #: Default
        "SuperFine",
        "TreutlerGM3",
        "TreutlerGM5",
    ]

    def _to_rex(self):
        return Template(
            """Some (exess_rex::XCGridResolution::Default
            exess_rex::XCGridDefaults::$default_grid
          )"""
        ).substitute(
            default_grid=self.default_grid,
        )


@dataclass
class CustomGridResolution:
    radial_size: int
    angular_size: int

    def _to_rex(self):
        return Template(
            """Some (exess_rex::XCGridResolution::Custom {
            radial_size = $radial_size,
            angular_size = $angular_size,
          })"""
        ).substitute(
            radial_size=self.radial_size,
            angular_size=self.angular_size,
        )


type XCGridResolutionT = DefaultGridResolution | CustomGridResolution


@dataclass
class ClosestAtomBatching:
    def _to_rex(self):
        return "Some (exess_rex::XCBatchingScheme::ClosestAtom)"


@dataclass
class Octree:
    max_size: int | None = None
    max_depth: int | None = None
    max_distance: float | None = None
    combine_small_children: bool | None = None

    def _rex_fields(self):
        return Template(
            """max_size = $maybe_max_size,
            max_depth = $maybe_max_depth,
            max_distance = $maybe_max_distance,
            combine_small_children = $maybe_combine_small_children,"""
        ).substitute(
            maybe_max_size=optional_str(self.max_size),
            maybe_max_depth=optional_str(self.max_depth),
            maybe_max_distance=optional_str(self.max_distance),
            maybe_combine_small_children=optional_str(self.combine_small_children),
        )

    def _to_rex(self):
        return Template(
            """Some (exess_rex::Octree {
            $fields
          })"""
        ).substitute(fields=self._rex_fields())


@dataclass
class OctreeBatching(Octree):
    def _to_rex(self):
        return Template(
            """Some (exess_rex::XCBatchingScheme::Octree {
            $fields
          })"""
        ).substitute(fields=self._rex_fields())


@dataclass
class SpaceFillingBatching:
    # TODO: fix
    octree: Octree | None = None
    target_batch_size: int | None = None

    def _to_rex(self):
        return Template(
            """Some (exess_rex::XCBatchingScheme::SpaceFilling {
            octree = $maybe_octree,
            target_batch_size = $maybe_target_batch_size,
          })"""
        ).substitute(
            maybe_octree=(self.octree._to_rex() if self.octree is not None else "None"),
            maybe_target_batch_size=optional_str(self.target_batch_size),
        )


@dataclass
class GauXCBatching:
    batch_size: int

    def _to_rex(self):
        return Template(
            """Some (exess_rex::XCBatchingScheme::GauXC {
            batch_size = $batch_size
          })"""
        ).substitute(
            batch_size=self.batch_size,
        )


type XCBatchingSchemeT = (
    ClosestAtomBatching | OctreeBatching | SpaceFillingBatching | GauXCBatching
)


@dataclass
class XCGridParameters:
    radial_quad: RadialQuadT | None = None
    pruning_scheme: PruningSchemeT | None = None
    consider_weight_zero: float | None = None
    resolution: XCGridResolutionT | None = None
    batching: XCBatchingSchemeT | None = None

    def _to_rex(self):
        return Template(
            """Some (exess_rex::XCGrid {
              radial_quad = $maybe_radial_quad,
              pruning_scheme = $maybe_pruning_scheme,
              consider_weight_zero = $maybe_consider_weight_zero,
              resolution = $maybe_resolution,
              batching = $maybe_batching,
            })"""
        ).substitute(
            maybe_radial_quad=optional_str(self.radial_quad, "exess_rex::RadialQuad::"),
            maybe_pruning_scheme=optional_str(
                self.pruning_scheme, "exess_rex::PruningScheme::"
            ),
            maybe_consider_weight_zero=optional_str(self.consider_weight_zero),
            maybe_resolution=(
                self.resolution._to_rex() if self.resolution is not None else "None"
            ),
            maybe_batching=(
                self.batching._to_rex() if self.batching is not None else "None"
            ),
        )


type KSDFTMethodT = Literal[
    "GauXC",  #: Upstream default, but we want BatchDense
    "Dense",
    "BatchDense",
    "Direct",
    "SemiDirect",
]


class _KSDFTDefault(enum.Enum):
    """Sentinel indicating that ksdft_keywords was not explicitly passed."""

    DEFAULT = enum.auto()


@dataclass
class KSDFTKeywords:
    """
    Configure runs done with the RestrictedKSDFT method.
    """

    #: KS-DFT functional to use
    functional: str
    grid: XCGridParameters | None = None
    method: KSDFTMethodT | None = "BatchDense"
    use_c_opt: bool | None = None
    sp_threshold: float | None = None
    dp_threshold: float | None = None
    batches_per_batch: int | None = None

    @staticmethod
    def resolve(
        ksdft_keywords: "KSDFTKeywords | _KSDFTDefault | None",
        method: str,
    ) -> "KSDFTKeywords | None":
        """Resolve ksdft_keywords default and warn if explicitly passed with a non-KSDFT method."""
        if isinstance(ksdft_keywords, _KSDFTDefault):
            return (
                KSDFTKeywords(functional="B3LYP")
                if method == "RestrictedKSDFT"
                else None
            )
        if ksdft_keywords is not None and method != "RestrictedKSDFT":
            warnings.warn(
                f"ksdft_keywords ignored: method is {method!r}, not 'RestrictedKSDFT'",
                stacklevel=3,
            )
            return None
        return ksdft_keywords

    def _to_rex(self):
        return Template(
            """Some (exess_rex::KSDFTKeywords {
            functional = "$functional",
            grid = $maybe_grid,
            method = $maybe_method,
            use_c_opt = $maybe_use_c_opt,
            sp_threshold = $maybe_sp_threshold,
            dp_threshold = $maybe_dp_threshold,
            batches_per_batch = $maybe_batches_per_batch,
          })"""
        ).substitute(
            functional=f"{self.functional}",
            maybe_grid=self.grid._to_rex() if self.grid is not None else "None",
            maybe_method=optional_str(self.method, "exess_rex::XCMethod::"),
            maybe_use_c_opt=optional_str(self.use_c_opt),
            maybe_sp_threshold=optional_str(self.sp_threshold),
            maybe_dp_threshold=optional_str(self.dp_threshold),
            maybe_batches_per_batch=optional_str(self.batches_per_batch),
        )


def exess(
    topology_path: Path | str,
    driver: str,
    method: MethodT = "RestrictedKSDFT",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords | None = FragKeywords(),
    ksdft_keywords: KSDFTKeywords | _KSDFTDefault | None = _KSDFTDefault.DEFAULT,
    export_keywords: ExportKeywords | None = None,
    system: System | None = None,
    convert_hdf5_to_json: bool | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the energy of the system in the QDX topology file at `topology_path`.
    """
    ksdft_keywords = KSDFTKeywords.resolve(ksdft_keywords, method)

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
        convert_hdf5_to_json = $maybe_convert_hdf5_to_json,
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
          ks_dft = $maybe_ks_keywords,
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
        maybe_convert_hdf5_to_json=optional_str(convert_hdf5_to_json),
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
        maybe_ks_keywords=(
            ksdft_keywords._to_rex() if ksdft_keywords is not None else "None"
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
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def exess_energy(
    topology_path: Path | str,
    method: MethodT = "RestrictedKSDFT",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords | None = FragKeywords(),
    ksdft_keywords: KSDFTKeywords | _KSDFTDefault | None = _KSDFTDefault.DEFAULT,
    export_keywords: ExportKeywords | None = None,
    system: System | None = None,
    convert_hdf5_to_json: bool | None = None,
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
        ksdft_keywords,
        export_keywords,
        system,
        convert_hdf5_to_json,
        run_spec,
        run_opts,
        collect,
    )


def _unwrap_outputs(
    res: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | RunError,
) -> tuple[dict[str, Any], dict[str, Any] | None] | RunError:
    if isinstance(res, RunError):
        return res

    if isinstance(res, dict):
        return (res, None)

    if len(res) == 0 or len(res) > 2:
        raise ValueError("exess_energy should return 1 or 2 outputs.")

    return (res[0], res[1] if len(res) == 2 else None)


def _resolve_exports_object(
    exports_obj: dict[str, Any] | None,
) -> tuple[Literal["Json", "Hdf5"], dict[str, Any]] | None:
    if exports_obj is None:
        return None
    if "Json" in exports_obj:
        return ("Json", exports_obj["Json"])
    if "Hdf5" in exports_obj:
        return ("Hdf5", exports_obj.get("Hdf5") or exports_obj)
    if "path" in exports_obj:
        if str(exports_obj.get("format", "")).lower() == "json":
            return ("Json", exports_obj)
        return ("Hdf5", exports_obj)
    raise ValueError(
        "Unknown output format in exports output. Expected 'Json' or 'Hdf5' key, "
        f"but got keys: {list(exports_obj.keys())}"
    )


def fetch_outputs(
    res: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | RunError,
    extract: bool = True,
) -> ExessResult | RunError:
    """
    Download EXESS outputs and convert the main JSON payload into Python dataclasses.

    Exported outputs are left lightly processed for now:
    - JSON exports are returned as a raw dict
    - HDF5 exports are returned as extracted file bytes by default
    - HDF5 exports are returned as raw tar.zst bytes when extract=False
    """
    outputs = _unwrap_outputs(res)
    if isinstance(outputs, RunError):
        return outputs

    output_obj, exports_obj = outputs

    calc = ExessCalculation.from_dict(
        json.loads(fetch_object(output_obj["path"]).decode())
    )

    exports: dict[str, Any] | bytes | None = None
    exports_ref = _resolve_exports_object(exports_obj)
    if exports_ref is not None:
        export_kind, exports_obj = exports_ref
        if export_kind == "Json":
            exports = json.loads(fetch_object(exports_obj["path"]).decode())
        elif export_kind == "Hdf5":
            try:
                exports = fetch_object(exports_obj["path"], extract=extract)
            except ValueError as e:
                if extract and "only directories" in str(e):
                    exports = None
                else:
                    raise
        else:
            raise ValueError(
                f"Unknown export kind {export_kind!r}. Expected 'Json' or 'Hdf5'."
            )

    return ExessResult(calc=calc, exports=exports)


def save_outputs(
    res: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | RunError,
    extract=True,
) -> ExessSavedResult | RunError:
    """
    Download and save energy calculation outputs from Rush.

    Takes the result from exess functions (`exess_energy()`, `exess_interaction_energy()`, etc.)
    and downloads the output files to disk. The result can be either a tuple or list
    because collect_run() returns a list when there are multiple outputs, which is
    converted to a tuple for consistent processing.

    Args:
        res: Result from exess calculation. Can be:
            - tuple[dict]: Single output from exess function
            - tuple[dict, dict]: Two outputs (JSON + HDF5)
            - list[dict]: Same as tuple (collect_run returns list for multiple outputs)
        extract: Whether to extract tar.zst files (default True)

    Returns:
        - ExessSavedResult(calc=..., exports=...): Local Paths to downloaded files
        - ExessSavedResult(calc=..., exports=None): If HDF5 extraction fails or no HDF5 output
        - RunError: If the input res is an error
    """
    outputs = _unwrap_outputs(res)
    if isinstance(outputs, RunError):
        return outputs

    output_obj, exports_obj = outputs
    calc_path = save_object(output_obj["path"])
    exports_ref = _resolve_exports_object(exports_obj)

    if exports_ref is None:
        return ExessSavedResult(calc=calc_path)

    exports_kind, exports_obj = exports_ref
    if exports_kind == "Json":
        return ExessSavedResult(
            calc=calc_path,
            exports=save_object(exports_obj["path"]),
        )
    elif exports_kind == "Hdf5":
        exports_path = None
        try:
            exports_path = save_object(
                exports_obj["path"],
                ext="hdf5" if extract else "tar.zst",
                extract=extract,
            )
        except ValueError as e:
            if "only directories" in str(e):
                # No actual files in HDF5 tar, return None for exports_path
                exports_path = None
            else:
                raise  # Re-raise other extraction errors
        return ExessSavedResult(calc=calc_path, exports=exports_path)

    raise ValueError(
        f"Unknown export kind {exports_kind!r}. Expected 'Json' or 'Hdf5'."
    )


def exess_interaction_energy(
    topology_path: Path | str,
    reference_fragment: int,
    method: MethodT = "RestrictedKSDFT",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords = FragKeywords(),
    ksdft_keywords: KSDFTKeywords | _KSDFTDefault | None = _KSDFTDefault.DEFAULT,
    system: System | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Compute the interaction energy between the fragment with index `reference_fragment` and the rest of the system
    in the toplogy file at `topology_path`.
    """
    return exess_energy(
        topology_path=topology_path,
        method=method,
        basis=basis,
        aux_basis=aux_basis,
        standard_orientation=standard_orientation,
        force_cartesian_basis_sets=force_cartesian_basis_sets,
        scf_keywords=scf_keywords,
        frag_keywords=replace(
            frag_keywords,
            reference_fragment=reference_fragment,
        ),
        ksdft_keywords=ksdft_keywords,
        system=system,
        run_spec=run_spec,
        run_opts=run_opts,
        collect=collect,
    )
