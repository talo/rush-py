#!/usr/bin/env python3
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template

import cyclopts
from gql.transport.exceptions import TransportQueryError

from .client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    collect_run,
    _submit_rex,
    upload_object,
)
from .utils import optional_str


@dataclass
class NnxtbResults:
    energy_mev: float
    forces_mev_per_angstrom: list[tuple[float, float, float]] | None
    frequencies_inv_cm: list[float] | None

    def __init__(
        self, energy_mev, forces_mev_per_angstrom=None, frequencies_inv_cm=None
    ):
        self.energy_mev = energy_mev
        self.forces_mev_per_angstrom = forces_mev_per_angstrom
        self.frequencies_inv_cm = frequencies_inv_cm


def nnxtb(
    topology_path: Path | str,
    compute_forces: bool | None = None,
    compute_frequencies: bool | None = None,
    multiplicity: int | None = None,
    run_spec: RunSpec = RunSpec(gpus=1, storage=100),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run nn-xTB on the system in the QDX topology file at `topology_path`.

    Returns the
    total energy, polar solvation energy, and nonpolar solvation energy
    of the system, in Hartrees.
    """

    # Upload inputs
    topology_vobj = upload_object(topology_path)
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
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def run_nnxtb():
    cyclopts.run(nnxtb)
