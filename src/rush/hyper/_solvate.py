"""Hyper solvation entrypoint wrapper."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from gql.transport.exceptions import TransportQueryError

from rush import TRC

from ..client import RunOpts, RunSpec, RushObject, _get_project_id, _submit_rex
from ..runs import Run as RushRun, RunID
from ._common import (
    ItemError,
    TRCInput,
    as_hyper_solvate_config,
    fetch_trc,
    parse_fallible_items,
    to_trc_vobj,
)


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to Hyper solvate outputs in the Rush object store."""

    items: list[RushObject | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ResultRef":
        parsed = parse_fallible_items(raw, lambda ok: RushObject.from_dict(ok))
        return cls(items=parsed)

    def fetch(self) -> list[TRC | ItemError]:
        return [item if isinstance(item, ItemError) else fetch_trc(item) for item in self.items]

    def save(self) -> list[Path | ItemError]:
        return [
            item if isinstance(item, ItemError) else item.save(ext="json")
            for item in self.items
        ]


def hyper_solvate_sumo(
    input_trcs: list[TRCInput],
    *,
    max_inputs: int | None = None,
    padding_nm: float | None = None,
    seed: int | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(gpus=0),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """Submit Hyper solvation jobs and return per-item TRC outputs."""

    input_vobjs = [to_trc_vobj(item) for item in input_trcs]
    inputs_expr = "[\n        " + ",\n        ".join(
        [f'obj_j "{obj["path"]}"' for obj in input_vobjs]
    ) + "]"

    config_expr = as_hyper_solvate_config(
        max_inputs=max_inputs,
        padding_nm=padding_nm,
        seed=seed,
        timeout_seconds=timeout_seconds,
    )

    rex = Template(
        """let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  hyper_solvate = λ structures →
    hyper_solvate_sumo_s
      ($run_spec)
      $config
      structures
in
  hyper_solvate $input_trcs
"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        config=config_expr,
        input_trcs=inputs_expr,
    )

    try:
        return RushRun(RunID(_submit_rex(_get_project_id(), rex, run_opts)), ResultRef)
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
