"""
NN-xTB module for the Rush Python client.

NN-xTB reparameterizes xTB with a neural network to approach DFT-level accuracy
while keeping xTB-like speed. It supports arbitrary charge and spin states and
is well-suited for large-scale screening where fast, per-atom forces or
vibrational frequencies are needed. Frequency calculations are more expensive.

Usage::

    from rush import nnxtb

    result = nnxtb.energy("mol.json").fetch()
    print(result.energy_mev)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from gql.transport.exceptions import TransportQueryError

from rush import TRC, Topology, TRCRef
from rush._trc import to_topology_vobj

from ._utils import optional_str
from .client import (
    RunOpts,
    RunSpec,
    RushObject,
    _get_project_id,
    _submit_rex,
    fetch_object,
)
from .run import RushRun

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """Parsed nn-xTB calculation results."""

    energy_mev: float
    forces_mev_per_angstrom: list[tuple[float, float, float]] | None = None
    frequencies_inv_cm: list[float] | None = None


@dataclass(frozen=True)
class ResultPaths:
    """Workspace path for saved nn-xTB output."""

    output: Path


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to nn-xTB output in the Rush object store."""

    output: RushObject

    @classmethod
    def from_raw_output(cls, res: Any) -> "ResultRef":
        """Parse raw ``collect_run`` output into a ``ResultRef``."""
        if not isinstance(res, list) or len(res) != 1:
            raise ValueError(
                f"nnxtb should return a list with exactly 1 output, "
                f"got {type(res).__name__}"
                f"{f' with {len(res)} items' if hasattr(res, '__len__') else ''}."
            )
        return cls(output=RushObject.from_dict(res[0]))

    def fetch(self) -> Result:
        """Download nn-xTB output and parse into Python objects."""
        return Result(**json.loads(fetch_object(self.output.path).decode()))

    def save(self) -> ResultPaths:
        """Download nn-xTB output and save to the workspace."""
        return ResultPaths(output=self.output.save())


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def energy(
    mol: TRC | TRCRef | Path | str | RushObject | Topology,
    compute_forces: bool | None = None,
    compute_frequencies: bool | None = None,
    multiplicity: int | None = None,
    run_spec: RunSpec = RunSpec(gpus=1, storage=100),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """
    Submit an nn-xTB energy calculation for the topology at *topology_path*.

    Returns a :class:`~rush.run.RushRun` handle. Call ``.fetch()`` to get the
    parsed result, or ``.save()`` to write it to disk.
    """

    # Upload inputs
    topology_vobj = to_topology_vobj(mol)
    charge = 0

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  nnxtb = λ topology →
    nnxtb_rex_s
      ($run_spec)
      (nnxtb_rex::NnxtbConfig {
        compute_forces = $maybe_compute_forces,
        compute_frequencies = $maybe_compute_frequencies,
        charge = $maybe_charge,
        multiplicity = $maybe_multiplicity,
      })
      (obj_j topology)
in
  nnxtb "$topology_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        maybe_compute_forces=optional_str(compute_forces),
        maybe_compute_frequencies=optional_str(compute_frequencies),
        maybe_charge=f"Some (int {charge})" if charge is not None else None,
        maybe_multiplicity=optional_str(multiplicity),
        topology_vobj_path=topology_vobj["path"],
    )
    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            ResultRef,
        )

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
