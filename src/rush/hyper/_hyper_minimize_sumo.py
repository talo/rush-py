"""Hyper minimization wrapper for the Rush Python client."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from string import Template

from gql.transport.exceptions import TransportQueryError

from .._rex import optional_str
from ..runs import Run, RunOpts, RunSpec
from ..session import _submit_rex
from ._common import (
    JsonObjectInput,
    TRCInput,
    _format_rex_list,
    _to_rex_json_obj,
    _upload_json_object,
    _upload_trc_object,
)
from ._hyper_solvate_sumo import TRCBatchResultRef


@dataclass(frozen=True)
class HyperMinimizeConfig:
    """Config for :func:`hyper_minimize_sumo`."""

    max_inputs: int | None = None
    steps: int | None = None
    gtol: float | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class MinimizeInput:
    """Input item for :func:`hyper_minimize_sumo`."""

    structure: TRCInput
    topology: JsonObjectInput


def _to_rex_minimize_config(config: HyperMinimizeConfig | None) -> str:
    if config is None:
        return "None"
    return Template(
        """Some (hyper_minimize_sumo::HyperMinimizeConfig {
    max_inputs = $max_inputs,
    steps = $steps,
    gtol = $gtol,
    timeout_seconds = $timeout_seconds,
  })"""
    ).substitute(
        max_inputs=optional_str(config.max_inputs),
        steps=optional_str(config.steps),
        gtol=optional_str(config.gtol),
        timeout_seconds=optional_str(config.timeout_seconds),
    )


def hyper_minimize_sumo(
    jobs: list[MinimizeInput],
    config: HyperMinimizeConfig | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> Run[TRCBatchResultRef]:
    """Submit Hyper minimization for one or more structures."""
    job_exprs: list[str] = []
    for job in jobs:
        structure = _to_rex_json_obj(_upload_trc_object(job.structure))
        topology = _to_rex_json_obj(_upload_json_object(job.topology))
        job_exprs.append(
            "("
            + "hyper_minimize_sumo::MinimizeInput { "
            + f"structure = {structure}, topology = {topology} "
            + "}"
            + ")"
        )

    rex = Template(
        """hyper_minimize_sumo_s
  ($run_spec)
  ($config)
  $jobs"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        config=_to_rex_minimize_config(config),
        jobs=_format_rex_list(job_exprs),
    )

    try:
        return Run(_submit_rex(rex, run_opts), TRCBatchResultRef)
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
