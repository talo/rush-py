#!/usr/bin/env python3
"""
EXESS QM/MM module helpers for the Rush Python client.

Run QM/MM simulations with EXESS.

Quick Links
-----------

- :func:`rush.exess_qmmm.exess_qmmm`
- :func:`rush.exess_qmmm.fetch_outputs`
- :func:`rush.exess_qmmm.save_outputs`
- :class:`rush.exess_qmmm.Trajectory`
- :class:`rush.exess_qmmm.Restraints`
- :mod:`rush.exess`
- :mod:`rush.exess_geo_opt`
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

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
    FragKeywords,
    KSDFTKeywords,
    MethodT,
    SCFKeywords,
    StandardOrientationT,
    System,
    _KSDFTDefault,
)
from .utils import optional_str


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


@dataclass
class ExessQMMMResult:
    geometries: list[list[float]]


def exess_qmmm(
    topology_path: Path | str,
    n_timesteps: int,
    residues_path: Path | str | None = None,
    dt_ps: float = 2e-3,
    temperature_kelvin: float = 290.0,
    pressure_atm: float | None = None,
    restraints: Restraints | None = None,
    trajectory: Trajectory = Trajectory(),
    gradient_finite_difference_step_size: float | None = None,
    method: MethodT = "RestrictedKSDFT",
    basis: BasisT = "cc-pVDZ",
    aux_basis: AuxBasisT | None = None,
    standard_orientation: StandardOrientationT | None = None,
    force_cartesian_basis_sets: bool | None = None,
    scf_keywords: SCFKeywords | None = None,
    frag_keywords: FragKeywords = FragKeywords(),
    ksdft_keywords: KSDFTKeywords | _KSDFTDefault | None = _KSDFTDefault.DEFAULT,
    qm_fragments: list[int] | None = None,
    mm_fragments: list[int] | None = None,
    system: System | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run a QMMM simulation of the system in the QDX topology and residues files at `topology_path` and `residues_path`.

    Specifying the number of timesteps is mandatory.
    If pressure is None, an NVT ensemble is used; if pressure is specified, an NPT ensemble is used.
    Fragments can be specified as QM or MM fragments via the respective parameters.
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
          ks_dft = $maybe_ks_keywords,
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
          machine_learning = None,
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
        maybe_ks_keywords=(
            ksdft_keywords._to_rex() if ksdft_keywords is not None else "None"
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


def fetch_outputs(
    res: dict[str, Any] | str | RunError,
) -> ExessQMMMResult | str | RunError:
    """
    Fetch EXESS QM/MM outputs into memory.
    """
    if isinstance(res, (str, RunError)):
        return res

    if not isinstance(res, dict) or "path" not in res:
        return RunError(
            f"Error: exess_qmmm output helper received unexpected format: {type(res)}"
        )

    return ExessQMMMResult(**json.loads(fetch_object(res["path"])))


def save_outputs(
    res: dict[str, Any] | str | RunError,
) -> Path | str | RunError:
    """
    Save EXESS QM/MM outputs into the workspace.
    """
    if isinstance(res, (str, RunError)):
        return res

    if not isinstance(res, dict) or "path" not in res:
        return RunError(
            f"Error: exess_qmmm output helper received unexpected format: {type(res)}"
        )

    return save_object(res["path"])


# TODO:
#  - trace for failure
#  - stdout, stderr
#  - other module instance info?
#  - qmmm minimisation config:
#    minimisation = Some (exess_rex::ClassicalMinimisation {
#      err_tol_kj_per_mol_nm = $err_tol_kj_per_mol_nm,
#      max_iterations = $max_iterations,
#    }),
