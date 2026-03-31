"""Hyper minimization entrypoint wrapper."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from gql.transport.exceptions import TransportQueryError

from rush import TRC

from ..client import RunOpts, RunSpec, RushObject, _get_project_id, _submit_rex
from ..run import RushRun
from ._common import (
    ItemError,
    JsonInput,
    TRCInput,
    as_hyper_minimize_config,
    fetch_trc,
    parse_fallible_items,
    to_json_vobj,
    to_trc_vobj,
)


@dataclass(frozen=True)
class MinimizeInput:
    """Single Hyper minimization job definition."""

    structure: TRCInput
    topology: JsonInput


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to Hyper minimize outputs in the Rush object store."""

    items: list[RushObject | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ResultRef":
        parsed = parse_fallible_items(raw, lambda ok: RushObject.from_dict(ok))
        return cls(items=parsed)

    def fetch(self) -> list[TRC | ItemError]:
        return [item if isinstance(item, ItemError) else fetch_trc(item) for item in self.items]

    def save(self) -> list[Path | ItemError]:
        return [item if isinstance(item, ItemError) else item.save(ext="json") for item in self.items]


def hyper_minimize_sumo(
    jobs: list[MinimizeInput],
    *,
    max_inputs: int | None = None,
    steps: int | None = None,
    gtol: float | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(gpus=0),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """Submit Hyper minimization jobs and return per-item TRC outputs."""

    input_exprs: list[str] = []
    for job in jobs:
        structure_vobj = to_trc_vobj(job.structure)
        topology_vobj = to_json_vobj(job.topology)
        input_exprs.append(
            "(hyper_minimize_sumo::MinimizeInput {"
            f" structure = (obj_j \"{structure_vobj['path']}\"),"
            f" topology = (obj_j \"{topology_vobj['path']}\")"
            " })"
        )

    jobs_expr = "[\n        " + ",\n        ".join(input_exprs) + "]"
    config_expr = as_hyper_minimize_config(
        max_inputs=max_inputs,
        steps=steps,
        gtol=gtol,
        timeout_seconds=timeout_seconds,
    )

    rex = Template(
        """let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  hyper_minimize = λ minimize_jobs →
    hyper_minimize_sumo_s
      ($run_spec)
      $config
      minimize_jobs
in
  hyper_minimize $jobs
"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        config=config_expr,
        jobs=jobs_expr,
    )

    try:
        return RushRun(_submit_rex(_get_project_id(), rex, run_opts), ResultRef)
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
