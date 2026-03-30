"""
EXESS QM/MM simulations for the Rush Python client.

Quick Links
-----------

- :func:`rush.exess.qmmm`
- :class:`rush.exess.QMMMResult`
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Self

from gql.transport.exceptions import TransportQueryError

from .._rex import optional_str
from ..mol import TRC, Residues, Topology
from ..objects import (
    RushObject,
    TRCRef,
    _to_residues_vobj,
    _to_topology_vobj,
)
from ..runs import Run, RunOpts, RunSpec
from ..session import _submit_rex
from ._energy import (
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

# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QMMMResult:
    geometries: list[list[float]]


@dataclass(frozen=True)
class QMMMResultPaths:
    output: Path


@dataclass(frozen=True)
class QMMMResultRef:
    """Lightweight reference to QM/MM outputs in the Rush object store."""

    output: RushObject

    @classmethod
    def from_raw_output(cls, res: Any) -> Self:
        """Parse raw ``collect_run`` output into a ``QMMMResultRef``."""
        if not isinstance(res, dict) or not isinstance(res.get("path"), str):
            raise ValueError(
                f"qmmm output received unexpected format: {type(res).__name__}"
            )
        return cls(output=RushObject.from_dict(res))

    def fetch(self) -> QMMMResult:
        """Download QM/MM outputs and parse into Python objects."""
        output = self.output.fetch_dict()
        return QMMMResult(**output)

    def save(self) -> QMMMResultPaths:
        """Download QM/MM outputs and save to the workspace."""
        return QMMMResultPaths(output=self.output.save())


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def qmmm(
    mol: TRC
    | TRCRef
    | tuple[Path | str | RushObject | Topology, Path | str | RushObject | Residues]
    | Path
    | str
    | RushObject
    | Topology,
    n_timesteps: int,
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
) -> Run[QMMMResultRef]:
    """
    Submit a QM/MM simulation for the topology at *topology_path*.

    Returns a :class:`~rush.runs.Run` handle. Call ``.fetch()`` to get the
    parsed trajectory, or ``.save()`` to write it to disk.
    """
    ksdft_keywords = KSDFTKeywords.resolve(ksdft_keywords, method)

    # Upload inputs
    residues_vobj = None
    match mol:
        case TRC() | TRCRef():
            topology_vobj = _to_topology_vobj(mol.topology)
            residues_vobj = _to_residues_vobj(mol.residues)
        case (t, r):
            topology_vobj = _to_topology_vobj(t)
            residues_vobj = _to_residues_vobj(r)
        case _:
            topology_vobj = _to_topology_vobj(mol)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology residues →
    try_exess_qmmm_rex
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
        return Run(_submit_rex(rex, run_opts), QMMMResultRef)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
