from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from gql.transport.exceptions import TransportQueryError

from ..mol import TRC

from .._utils import optional_str
from ..client import (
    RunOpts,
    RunSpec,
    RushObject,
    _get_project_id,
    _submit_rex,
    fetch_object,
)
from ..convert import from_json
from ..run import RushRun
from ._common import (
    ItemError,
    parse_fallible_items,
    topology_input_to_vobj,
    trc_object_input_to_vobj,
)


HyperTopologyInput = Path | str | RushObject | dict[str, Any]


@dataclass(frozen=True)
class MinimizeInput:
    structure: TRC | Path | str | RushObject
    topology: HyperTopologyInput


@dataclass(frozen=True)
class MinimizeOutputRef:
    """Reference to one successful `hyper_minimize_sumo` output object."""

    output: RushObject

    @classmethod
    def from_raw(cls, raw: Any) -> "MinimizeOutputRef":
        if not isinstance(raw, dict):
            raise ValueError(
                f"Minimize output object should be a dict, got {type(raw).__name__}."
            )
        return cls(output=RushObject.from_dict(raw))

    def fetch(self) -> TRC:
        payload = fetch_object(self.output.path)
        decoded = payload.decode() if isinstance(payload, bytes) else payload
        return from_json(json.loads(decoded))

    def save(self) -> Path:
        return self.output.save(ext="json")


@dataclass(frozen=True)
class ResultRef:
    """Collected output for `hyper_minimize_sumo`."""

    items: list[MinimizeOutputRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ResultRef":
        return cls(
            items=parse_fallible_items(
                raw,
                parse_ok=MinimizeOutputRef.from_raw,
            )
        )

    def __getitem__(self, index: int) -> MinimizeOutputRef | ItemError:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[MinimizeOutputRef | ItemError]:
        return iter(self.items)

    def fetch(self) -> list[TRC | ItemError]:
        return [item.fetch() if isinstance(item, MinimizeOutputRef) else item for item in self.items]

    def save(self) -> list[Path | ItemError]:
        return [item.save() if isinstance(item, MinimizeOutputRef) else item for item in self.items]


def hyper_minimize_sumo(
    jobs: list[MinimizeInput],
    *,
    max_inputs: int | None = None,
    steps: int | None = None,
    gtol: float | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """Run Hyper minimization for one or more structure/topology jobs."""

    job_exprs: list[str] = []
    for job in jobs:
        structure_vobj = trc_object_input_to_vobj(job.structure)
        topology_vobj = topology_input_to_vobj(job.topology)
        job_exprs.append(
            """(hyper_minimize_sumo::MinimizeInput {
            structure = (obj_j "$structure"),
            topology = (obj_j "$topology")
          })""".replace("$structure", structure_vobj["path"]).replace(
                "$topology", topology_vobj["path"]
            )
        )

    config_expr = (
        "None"
        if all(v is None for v in (max_inputs, steps, gtol, timeout_seconds))
        else """Some (hyper_minimize_sumo::HyperMinimizeConfig {
        max_inputs = $max_inputs,
        steps = $steps,
        gtol = $gtol,
        timeout_seconds = $timeout_seconds,
      })"""
    )

    if config_expr != "None":
        config_expr = Template(config_expr).substitute(
            max_inputs=optional_str(max_inputs),
            steps=optional_str(steps),
            gtol=optional_str(gtol),
            timeout_seconds=optional_str(timeout_seconds),
        )

    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  run = λ jobs →
    hyper_minimize_sumo_s
      ($run_spec)
      ($config)
      jobs
in
  run [$jobs]
""").substitute(
        run_spec=run_spec._to_rex(),
        config=config_expr,
        jobs=", ".join(job_exprs),
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
