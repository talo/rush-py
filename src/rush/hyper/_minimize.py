from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from gql.transport.exceptions import TransportQueryError

from rush import TRC

from .._utils import optional_str
from ..client import RunOpts, RunSpec, RushObject, _get_project_id, _submit_rex
from ..run import RushRun
from ._shared import (
    HyperTopologyInput,
    ItemError,
    TRCInput,
    fetch_trc_object,
    parse_item_results,
    to_hyper_topology_vobj,
    to_trc_vobj,
)


@dataclass
class HyperMinimizeConfig:
    max_inputs: int | None = None
    steps: int | None = None
    gtol: float | None = None
    timeout_seconds: int | None = None

    def _to_rex(self) -> str:
        return Template(
            """Some (hyper_minimize_sumo::HyperMinimizeConfig {
        max_inputs = $max_inputs,
        steps = $steps,
        gtol = $gtol,
        timeout_seconds = $timeout_seconds,
      })"""
        ).substitute(
            max_inputs=optional_str(self.max_inputs),
            steps=optional_str(self.steps),
            gtol=optional_str(self.gtol),
            timeout_seconds=optional_str(self.timeout_seconds),
        )


@dataclass(frozen=True)
class MinimizeInput:
    structure: TRCInput
    topology: HyperTopologyInput


@dataclass(frozen=True)
class MinimizeResultRef:
    """Per-input result references for `hyper_minimize_sumo`."""

    _items: list[RushObject | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "MinimizeResultRef":
        return cls(
            _items=parse_item_results(
                raw,
                module_name="hyper_minimize_sumo",
                on_success=RushObject.from_dict,
            )
        )

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[RushObject | ItemError]:
        return iter(self._items)

    def __getitem__(self, index: int) -> RushObject | ItemError:
        return self._items[index]

    def fetch(self) -> list[TRC | ItemError]:
        return [
            fetch_trc_object(item) if isinstance(item, RushObject) else item
            for item in self._items
        ]

    def save(self) -> list[Path | ItemError]:
        return [
            item.save(ext="json") if isinstance(item, RushObject) else item
            for item in self._items
        ]


def hyper_minimize_sumo(
    jobs: Sequence[MinimizeInput],
    config: HyperMinimizeConfig | None = None,
    run_spec: RunSpec = RunSpec(storage=4096),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[MinimizeResultRef]:
    """Run Hyper minimization for one or more structure/topology job pairs."""

    job_exprs: list[str] = []
    for job in jobs:
        structure_vobj = to_trc_vobj(job.structure)
        topology_vobj = to_hyper_topology_vobj(job.topology)
        job_exprs.append(
            Template(
                """(hyper_minimize_sumo::MinimizeInput {
          structure = (obj_j \"$structure_path\"),
          topology = (obj_j \"$topology_path\"),
        })"""
            ).substitute(
                structure_path=structure_vobj["path"],
                topology_path=topology_vobj["path"],
            )
        )

    jobs_expr = (
        "[]"
        if not job_exprs
        else "[\n        " + ",\n        ".join(job_exprs) + "\n      ]"
    )

    rex = Template(
        """let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 }
in
  hyper_minimize_sumo_s
    ($run_spec)
    ($maybe_config)
    $jobs
"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        maybe_config=config._to_rex() if config is not None else "None",
        jobs=jobs_expr,
    )

    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            MinimizeResultRef,
        )
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
