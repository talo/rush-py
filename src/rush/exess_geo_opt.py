#!/usr/bin/env python3
"""
EXESS geometry optimization helpers for the Rush Python client.

Run geometry optimization with EXESS.

Quick Links
-----------

- :func:`rush.exess_geo_opt.exess_geo_opt`
- :func:`rush.exess_geo_opt.fetch_outputs`
- :func:`rush.exess_geo_opt.save_outputs`
- :mod:`rush.exess`
- :mod:`rush.exess_qmmm`
"""

import json
import sys
from dataclasses import dataclass
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
from .exess import (
    AuxBasisT,
    BasisT,
    KSDFTKeywords,
    MethodT,
    SCFKeywords,
    StandardOrientationT,
    System,
    _KSDFTDefault,
)
from .mol import Topology
from .utils import optional_str


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


@dataclass
class ExessGeoOptStep:
    total_energy: float | None = None
    max_gradient_component: float | None = None


@dataclass
class ExessGeoOptResult:
    trajectory: list[Topology]
    steps: list[ExessGeoOptStep]


def exess_geo_opt(
    topology_path: Path | str,
    max_iters: int,
    residues_path: Path | str | None = None,
    optimization_keywords: OptimizationKeywords = OptimizationKeywords(),
    method: MethodT = "RestrictedKSDFT",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    ksdft_keywords: KSDFTKeywords | _KSDFTDefault | None = _KSDFTDefault.DEFAULT,
    qm_fragments: list[int] | None = None,
    mm_fragments: list[int] | None = None,
    system: System | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run optimization on the system in the QDX topology and residues files at `topology_path`.

    Specifying the maximum iterations is mandatory.
    Fragment-based QM calculation is not supported, but fragments can be used for specifying regions as QM or MM.
    If one fragment list parameter is specified, the rest of the fragments are inferred to be of the other type.
    If both fragment list parameters are specified, each fragment must be placed in exactly one of the lists.
    """
    ksdft_keywords = KSDFTKeywords.resolve(ksdft_keywords, method)

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
          ks_dft = $maybe_ks_keywords,
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
          qmmm = $maybe_qmmm_keywords,
          machine_learning = None,
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
        maybe_ks_keywords=(
            ksdft_keywords._to_rex() if ksdft_keywords is not None else "None"
        ),
        maybe_optimization_keywords=(
            optimization_keywords._to_rex(max_iters)
            if optimization_keywords is not None
            else "None"
        ),
        maybe_qmmm_keywords=(
            """Some (exess_qmmm_rex::QMMMKeywords {
            n_timesteps = 1,
            dt_ps = 0.002,
            temperature_kelvin = 290.0,
            pressure_atm = None,
            minimisation = None,
            trajectory = None,
            restraints = None,
            energy_csv = None,
          })"""
            if mm_fragments or (qm_fragments is not None)
            else "None"
        ),
        maybe_regions=(
            Template(
                """Some (exess_qmmm_rex::RegionKeywords {
            qm_fragments = $maybe_qm_fragments,
            mm_fragments = $maybe_mm_fragments,
            ml_fragments = Some [],
          })"""
            ).substitute(
                maybe_qm_fragments=optional_str(qm_fragments),
                maybe_mm_fragments=optional_str(mm_fragments),
            )
            if not (qm_fragments is None and mm_fragments is None)
            else "None"
        ),
        residues_expr=(
            "(Some [ (obj_j residues) ])" if residues_path is not None else "None"
        ),
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"] if residues_vobj is not None else "",
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


def _unwrap_outputs(
    res: list[dict[str, Any]] | tuple[dict[str, Any], ...] | str | RunError,
) -> tuple[dict[str, Any], dict[str, Any]] | str | RunError:
    if isinstance(res, (str, RunError)):
        return res

    if isinstance(res, (list, tuple)) and len(res) == 2:
        return (res[0], res[1])

    return RunError(
        f"Error: exess_geo_opt output helper received unexpected format: {type(res)}"
    )


def fetch_outputs(
    res: list[dict[str, Any]] | tuple[dict[str, Any], ...] | str | RunError,
) -> ExessGeoOptResult | str | RunError:
    """
    Fetch EXESS geometry optimization outputs into memory.
    """
    outputs = _unwrap_outputs(res)
    if isinstance(outputs, (str, RunError)):
        return outputs

    trajectory_obj, steps_obj = outputs
    trajectory = [
        Topology.from_json(topology)
        for topology in json.loads(fetch_object(trajectory_obj["path"]))
    ]
    steps = [
        ExessGeoOptStep(**step) for step in json.loads(fetch_object(steps_obj["path"]))
    ]
    return ExessGeoOptResult(trajectory=trajectory, steps=steps)


def save_outputs(
    res: list[dict[str, Any]] | tuple[dict[str, Any], ...] | str | RunError,
) -> tuple[Path, Path] | str | RunError:
    """
    Save EXESS geometry optimization outputs into the workspace.
    """
    outputs = _unwrap_outputs(res)
    if isinstance(outputs, (str, RunError)):
        return outputs

    trajectory_obj, steps_obj = outputs
    return (
        save_object(trajectory_obj["path"]),
        save_object(steps_obj["path"]),
    )
