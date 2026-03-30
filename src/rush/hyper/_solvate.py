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

from .._rex import optional_str
from ..convert import from_json
from ..objects import RushObject, fetch_object
from ..runs import Run, RunOpts, RunSpec
from ..session import _submit_rex
from ._common import ItemError, parse_fallible_items, trc_object_input_to_vobj


@dataclass(frozen=True)
class SolvateOutputRef:
    """Reference to one successful `hyper_solvate_sumo` output object."""

    output: RushObject

    @classmethod
    def from_raw(cls, raw: Any) -> "SolvateOutputRef":
        if not isinstance(raw, dict):
            raise ValueError(
                f"Solvate output object should be a dict, got {type(raw).__name__}."
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
    """Collected output for `hyper_solvate_sumo`."""

    items: list[SolvateOutputRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ResultRef":
        return cls(
            items=parse_fallible_items(
                raw,
                parse_ok=SolvateOutputRef.from_raw,
            )
        )

    def __getitem__(self, index: int) -> SolvateOutputRef | ItemError:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[SolvateOutputRef | ItemError]:
        return iter(self.items)

    def fetch(self) -> list[TRC | ItemError]:
        return [item.fetch() if isinstance(item, SolvateOutputRef) else item for item in self.items]

    def save(self) -> list[Path | ItemError]:
        return [item.save() if isinstance(item, SolvateOutputRef) else item for item in self.items]


def hyper_solvate_sumo(
    input_trcs: list[TRC | Path | str | RushObject],
    *,
    max_inputs: int | None = None,
    padding_nm: float | None = None,
    seed: int | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> Run[ResultRef]:
    """Run Hyper solvation and return one output per input item."""

    input_exprs: list[str] = []
    for item in input_trcs:
        vobj = trc_object_input_to_vobj(item)
        input_exprs.append(f'(obj_j "{vobj["path"]}")')

    config_expr = (
        "None"
        if all(v is None for v in (max_inputs, padding_nm, seed, timeout_seconds))
        else """Some (hyper_solvate_sumo::HyperConfig {
        max_inputs = $max_inputs,
        padding_nm = $padding_nm,
        seed = $seed,
        timeout_seconds = $timeout_seconds,
      })"""
    )

    if config_expr != "None":
        config_expr = Template(config_expr).substitute(
            max_inputs=optional_str(max_inputs),
            padding_nm=optional_str(padding_nm),
            seed=optional_str(seed),
            timeout_seconds=optional_str(timeout_seconds),
        )

    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  run = λ inputs →
    hyper_solvate_sumo_s
      ($run_spec)
      ($config)
      inputs
in
  run [$inputs]
""").substitute(
        run_spec=run_spec._to_rex(),
        config=config_expr,
        inputs=", ".join(input_exprs),
    )

    try:
        return Run(
            _submit_rex(rex, run_opts),
            ResultRef,
        )
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
