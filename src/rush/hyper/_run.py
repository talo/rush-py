"""Hyper MD run entrypoint wrapper."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from string import Template
from typing import Any, Literal

from gql.transport.exceptions import TransportQueryError

from ..client import RunOpts, RunSpec, RushObject, _get_project_id, _submit_rex
from ..runs import Run as RushRun
from ._common import (
    HyperRunOutput,
    HyperRunOutputPaths,
    HyperRunOutputRef,
    ItemError,
    JsonInput,
    TRCInput,
    as_hyper_run_config,
    fetch_bytes,
    parse_fallible_items,
    to_json_vobj,
    to_trc_vobj,
)


@dataclass(frozen=True)
class RunInput:
    """Single Hyper run job definition."""

    sim_config: JsonInput
    topology: JsonInput
    coordinates: TRCInput


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to Hyper run outputs in the Rush object store."""

    items: list[HyperRunOutputRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ResultRef":
        def parse_ok(ok_payload: Any) -> HyperRunOutputRef:
            if not isinstance(ok_payload, dict):
                raise ValueError(
                    f"Run output item should be a dict, got {type(ok_payload).__name__}."
                )
            if "trajectory" not in ok_payload:
                raise ValueError("Run output item missing required key 'trajectory'.")

            trajectory = RushObject.from_dict(ok_payload["trajectory"])
            checkpoint = (
                RushObject.from_dict(ok_payload["checkpoint"])
                if ok_payload.get("checkpoint") is not None
                else None
            )
            return HyperRunOutputRef(trajectory=trajectory, checkpoint=checkpoint)

        parsed = parse_fallible_items(raw, parse_ok)
        return cls(items=parsed)

    def fetch(self) -> list[HyperRunOutput | ItemError]:
        outputs: list[HyperRunOutput | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                outputs.append(item)
                continue

            outputs.append(
                HyperRunOutput(
                    trajectory=fetch_bytes(item.trajectory),
                    checkpoint=(
                        fetch_bytes(item.checkpoint) if item.checkpoint is not None else None
                    ),
                )
            )
        return outputs

    def save(self) -> list[HyperRunOutputPaths | ItemError]:
        outputs: list[HyperRunOutputPaths | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                outputs.append(item)
                continue

            outputs.append(
                HyperRunOutputPaths(
                    trajectory=item.trajectory.save(ext="xtc"),
                    checkpoint=(
                        item.checkpoint.save(ext="bin") if item.checkpoint is not None else None
                    ),
                )
            )
        return outputs


def hyper_run_sumo(
    jobs: list[RunInput],
    *,
    max_inputs: int | None = None,
    nsteps: int | None = None,
    dt_ps: float | None = None,
    temperature_k: float | None = None,
    ensemble: Literal["Nve", "Nvt", "Npt"] | None = None,
    minimize_before_run: bool | None = None,
    solvate_before_run: bool | None = None,
    use_gpu: bool | None = None,
    nthreads: int | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(gpus=0),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """Submit Hyper molecular dynamics jobs and return per-item run artifacts."""

    job_exprs: list[str] = []
    for job in jobs:
        sim_config_vobj = to_json_vobj(job.sim_config)
        topology_vobj = to_json_vobj(job.topology)
        coordinates_vobj = to_trc_vobj(job.coordinates)

        job_exprs.append(
            "(hyper_run_sumo::RunInput {"
            f" sim_config = (obj_j \"{sim_config_vobj['path']}\"),"
            f" topology = (obj_j \"{topology_vobj['path']}\"),"
            f" coordinates = (obj_j \"{coordinates_vobj['path']}\")"
            " })"
        )

    jobs_expr = "[\n        " + ",\n        ".join(job_exprs) + "]"
    config_expr = as_hyper_run_config(
        max_inputs=max_inputs,
        nsteps=nsteps,
        dt_ps=dt_ps,
        temperature_k=temperature_k,
        ensemble=ensemble,
        minimize_before_run=minimize_before_run,
        solvate_before_run=solvate_before_run,
        use_gpu=use_gpu,
        nthreads=nthreads,
        timeout_seconds=timeout_seconds,
    )

    rex = Template(
        """let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  hyper_run = λ run_jobs →
    hyper_run_sumo_s
      ($run_spec)
      $config
      run_jobs
in
  hyper_run $jobs
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
