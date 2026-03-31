"""Hyper solvation wrapper for the Rush Python client."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Self

from gql.transport.exceptions import TransportQueryError

from .._rex import optional_str
from ..mol import TRC
from ..objects import RushObject
from ..runs import Run, RunOpts, RunSpec
from ..session import _submit_rex
from ._common import (
    ItemError,
    TRCInput,
    _fetch_trc,
    _format_rex_list,
    _parse_batch_outputs,
    _to_rex_json_obj,
    _upload_trc_object,
)


@dataclass(frozen=True)
class HyperConfig:
    """Config for :func:`hyper_solvate_sumo`."""

    max_inputs: int | None = None
    padding_nm: float | None = None
    seed: int | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class TRCBatchResultRef:
    """Result reference for TRC-producing Hyper batch entrypoints."""

    items: list[RushObject | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> Self:
        parsed = _parse_batch_outputs(raw, _parse_trc_item, "hyper TRC batch")
        return cls(items=parsed)

    def __getitem__(self, index: int) -> RushObject | ItemError:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def fetch(self) -> list[TRC | ItemError]:
        out: list[TRC | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                out.append(item)
            else:
                out.append(_fetch_trc(item))
        return out

    def save(self) -> list[Path | ItemError]:
        return [
            item if isinstance(item, ItemError) else item.save(ext="json")
            for item in self.items
        ]


def _parse_trc_item(raw: Any) -> RushObject:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected TRC output object, got {type(raw).__name__}")
    return RushObject.from_dict(raw)

def _to_rex_solvate_config(config: HyperConfig | None) -> str:
    if config is None:
        return "None"
    return Template(
        """Some (hyper_solvate_sumo::HyperConfig {
    max_inputs = $max_inputs,
    padding_nm = $padding_nm,
    seed = $seed,
    timeout_seconds = $timeout_seconds,
  })"""
    ).substitute(
        max_inputs=optional_str(config.max_inputs),
        padding_nm=optional_str(config.padding_nm),
        seed=optional_str(config.seed),
        timeout_seconds=optional_str(config.timeout_seconds),
    )


def hyper_solvate_sumo(
    input_trcs: list[TRCInput],
    config: HyperConfig | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> Run[TRCBatchResultRef]:
    """Submit Hyper solvation for one or more TRC inputs."""
    input_exprs = [_to_rex_json_obj(_upload_trc_object(item)) for item in input_trcs]

    rex = Template(
        """hyper_solvate_sumo_s
  ($run_spec)
  ($config)
  $inputs"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        config=_to_rex_solvate_config(config),
        inputs=_format_rex_list(input_exprs),
    )

    try:
        return Run(_submit_rex(rex, run_opts), TRCBatchResultRef)
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
