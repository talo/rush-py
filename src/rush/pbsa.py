"""
PBSA module for the Rush Python client.

Computes solvation energies using the Poisson-Boltzmann Surface Area method.

Usage::

    from rush import pbsa

    result = pbsa.solvation_energy("mol.json", ...).fetch()
    print(result.solvation_energy)
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Any, Self

from gql.transport.exceptions import TransportQueryError

from ._rex import float_to_str
from .mol import TRC, Topology
from .objects import (
    RushObject,
    TRCRef,
    _json_content_name,
    _to_topology_vobj,
    save_json,
)
from .runs import Run, RunOpts, RunSpec
from .session import _submit_rex

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """Parsed PBSA solvation energy results (all values in Hartrees)."""

    solvation_energy: float
    polar_solvation_energy: float
    nonpolar_solvation_energy: float


@dataclass(frozen=True)
class ResultPaths:
    """Workspace path for saved PBSA output."""

    output: Path


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to PBSA output.

    PBSA results are small enough to be returned inline (three floats),
    so no object store download is needed.
    """

    solvation_energy: float
    polar_solvation_energy: float
    nonpolar_solvation_energy: float

    @classmethod
    def from_raw_output(cls, res: Any) -> Self:
        """Parse raw ``collect_run`` output into a ``ResultRef``."""
        if isinstance(res, list) and len(res) == 3:
            return cls(
                solvation_energy=float(res[0]),
                polar_solvation_energy=float(res[1]),
                nonpolar_solvation_energy=float(res[2]),
            )
        raise ValueError(
            f"pbsa should return exactly 3 float outputs, "
            f"got {type(res).__name__} with {len(res) if hasattr(res, '__len__') else '?'} items."
        )

    def fetch(self) -> Result:
        """Return parsed PBSA results (no download needed — data is inline)."""
        return Result(
            solvation_energy=self.solvation_energy,
            polar_solvation_energy=self.polar_solvation_energy,
            nonpolar_solvation_energy=self.nonpolar_solvation_energy,
        )

    def save(self) -> ResultPaths:
        """Save PBSA results as JSON to the workspace."""
        output_json = asdict(self.fetch())
        return ResultPaths(
            output=save_json(
                output_json,
                name=_json_content_name("pbsa_output", output_json),
            ),
        )


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def solvation_energy(
    mol: TRC | TRCRef | Path | str | RushObject | Topology,
    solute_dielectric: float,
    solvent_dielectric: float,
    solvent_radius: float,
    ion_concentration: float,
    temperature: float,
    spacing: float,
    sasa_gamma: float,
    sasa_beta: float,
    sasa_n_samples: int,
    convergence: float,
    box_size_factor: float,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
) -> Run[ResultRef]:
    """
    Submit a PBSA solvation energy calculation for the topology at *topology_path*.

    Returns a :class:`~rush.runs.Run` handle. Call ``.fetch()`` to get the
    parsed result, or ``.save()`` to write it to disk as JSON.
    """

    # Upload inputs
    topology_vobj = _to_topology_vobj(mol)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  pbsa = λ topology →
    try_pbsa_rex
      ($run_spec)
      (pbsa_rex::PBSAParameters {
        solute_dielectric = $solute_dielectric,
        solvent_dielectric = $solvent_dielectric,
        solvent_radius = $solvent_radius,
        ion_concentration = $ion_concentration,
        temperature = $temperature,
        spacing = $spacing,
        sasa_gamma = $sasa_gamma,
        sasa_beta = $sasa_beta,
        sasa_n_samples = $sasa_n_samples,
        convergence = $convergence,
        box_size_factor = $box_size_factor,
      })
      (obj_j topology)
in
  pbsa "$topology_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        solute_dielectric=float_to_str(solute_dielectric),
        solvent_dielectric=float_to_str(solvent_dielectric),
        solvent_radius=float_to_str(solvent_radius),
        ion_concentration=float_to_str(ion_concentration),
        temperature=float_to_str(temperature),
        spacing=float_to_str(spacing),
        sasa_gamma=float_to_str(sasa_gamma),
        sasa_beta=float_to_str(sasa_beta),
        sasa_n_samples=sasa_n_samples,
        convergence=float_to_str(convergence),
        box_size_factor=float_to_str(box_size_factor),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        return Run(_submit_rex(rex, run_opts), ResultRef)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
