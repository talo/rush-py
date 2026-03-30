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
from ._shared import ItemError, TRCInput, fetch_trc_object, parse_item_results, to_trc_vobj


@dataclass
class HyperConfig:
    max_inputs: int | None = None
    padding_nm: float | None = None
    seed: int | None = None
    timeout_seconds: int | None = None

    def _to_rex(self) -> str:
        return Template(
            """Some (hyper_solvate_sumo::HyperConfig {
        max_inputs = $max_inputs,
        padding_nm = $padding_nm,
        seed = $seed,
        timeout_seconds = $timeout_seconds,
      })"""
        ).substitute(
            max_inputs=optional_str(self.max_inputs),
            padding_nm=optional_str(self.padding_nm),
            seed=optional_str(self.seed),
            timeout_seconds=optional_str(self.timeout_seconds),
        )


@dataclass(frozen=True)
class SolvateResultRef:
    """Per-input result references for `hyper_solvate_sumo`."""

    _items: list[RushObject | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "SolvateResultRef":
        return cls(
            _items=parse_item_results(
                raw,
                module_name="hyper_solvate_sumo",
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


def hyper_solvate_sumo(
    input_trcs: Sequence[TRCInput],
    config: HyperConfig | None = None,
    run_spec: RunSpec = RunSpec(storage=4096),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[SolvateResultRef]:
    """Run Hyper solvation for one or more TRC inputs."""

    input_vobjs = [to_trc_vobj(trc) for trc in input_trcs]
    inputs_expr = (
        "[]"
        if not input_vobjs
        else "[\n        "
        + ",\n        ".join(f'(obj_j "{obj["path"]}")' for obj in input_vobjs)
        + "\n      ]"
    )

    rex = Template(
        """let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 }
in
  hyper_solvate_sumo_s
    ($run_spec)
    ($maybe_config)
    $inputs
"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        maybe_config=config._to_rex() if config is not None else "None",
        inputs=inputs_expr,
    )

    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            SolvateResultRef,
        )
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
