#!/usr/bin/env python3
"""
NN-xTB module helpers for the Rush Python client.

NN-xTB reparameterizes xTB with a neural network to approach DFT-level accuracy
while keeping xTB-like speed. It supports arbitrary charge and spin states and
is well-suited for large-scale screening where fast, per-atom forces or
vibrational frequencies are needed. Frequency calculations are more expensive.
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
from .utils import optional_str


@dataclass
class NnxtbResult:
    """
    Parsed nn-xTB results.

    Use `fetch_outputs(nnxtb(..., collect=True))` to return this dataclass in
    memory, or `save_outputs(nnxtb(..., collect=True))` to save the raw JSON
    output into the workspace.
    """

    energy_mev: float
    forces_mev_per_angstrom: list[tuple[float, float, float]] | None = None
    frequencies_inv_cm: list[float] | None = None


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
    Run NN-xTB on the system in the QDX topology file at `topology_path`.

    Args:
        topology_path: Path to a TRC topology JSON file.
        compute_forces: Whether to compute per-atom forces.
            Defaults to true.
        compute_frequencies: Whether to compute vibrational frequencies.
            Defaults to false.
        multiplicity: Spin multiplicity. Defaults to 1 (singlet).
        run_spec: Rush compute resources to request.
        run_opts: Rush run metadata.
        collect: Whether to wait for completion and return outputs.
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
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def _unwrap_output(
    res: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | RunError,
) -> dict[str, Any] | RunError:
    if isinstance(res, RunError):
        return res
    if isinstance(res, dict):
        return res
    if len(res) != 1:
        raise ValueError("nnxtb should return exactly 1 output.")
    return res[0]


def fetch_outputs(
    res: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | RunError,
) -> NnxtbResult | RunError:
    output_obj = _unwrap_output(res)
    if isinstance(output_obj, RunError):
        return output_obj
    return NnxtbResult(**json.loads(fetch_object(output_obj["path"]).decode()))


def save_outputs(
    res: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...] | RunError,
) -> Path | RunError:
    output_obj = _unwrap_output(res)
    if isinstance(output_obj, RunError):
        return output_obj
    return save_object(output_obj["path"])
