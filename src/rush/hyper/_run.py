from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Literal

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
from ..run import RushRun
from ._common import (
    ItemError,
    parse_fallible_items,
    sim_config_input_to_vobj,
    topology_input_to_vobj,
    trc_object_input_to_vobj,
)


HyperTopologyInput = Path | str | RushObject | dict[str, Any]
Ensemble = Literal["Nve", "Nvt", "Npt"]


@dataclass(frozen=True)
class RunInput:
    sim_config_json: Path | str | RushObject
    topology: HyperTopologyInput
    coordinates: TRC | Path | str | RushObject


@dataclass(frozen=True)
class RunOutputRef:
    """Reference to one successful `hyper_run_sumo` output item."""

    trajectory: RushObject
    checkpoint: RushObject | None

    @classmethod
    def from_raw(cls, raw: Any) -> "RunOutputRef":
        if not isinstance(raw, dict):
            raise ValueError(f"Run output should be a dict, got {type(raw).__name__}.")

        trajectory = raw.get("trajectory")
        if not isinstance(trajectory, dict):
            raise ValueError("Run output missing required 'trajectory' object.")

        checkpoint_raw = raw.get("checkpoint")
        checkpoint = None
        if checkpoint_raw is not None:
            if not isinstance(checkpoint_raw, dict):
                raise ValueError("Run output 'checkpoint' must be an object or null.")
            checkpoint = RushObject.from_dict(checkpoint_raw)

        return cls(
            trajectory=RushObject.from_dict(trajectory),
            checkpoint=checkpoint,
        )


@dataclass(frozen=True)
class RunOutput:
    trajectory: bytes
    checkpoint: bytes | None


@dataclass(frozen=True)
class RunOutputPaths:
    trajectory: Path
    checkpoint: Path | None


@dataclass(frozen=True)
class ResultRef:
    """Collected output for `hyper_run_sumo`."""

    items: list[RunOutputRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ResultRef":
        return cls(items=parse_fallible_items(raw, parse_ok=RunOutputRef.from_raw))

    def __getitem__(self, index: int) -> RunOutputRef | ItemError:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[RunOutputRef | ItemError]:
        return iter(self.items)

    def fetch(self) -> list[RunOutput | ItemError]:
        output: list[RunOutput | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                output.append(item)
                continue

            trajectory_payload = fetch_object(item.trajectory.path)
            checkpoint_payload = (
                fetch_object(item.checkpoint.path)
                if item.checkpoint is not None
                else None
            )

            output.append(
                RunOutput(
                    trajectory=(
                        trajectory_payload
                        if isinstance(trajectory_payload, bytes)
                        else trajectory_payload.encode()
                    ),
                    checkpoint=(
                        checkpoint_payload
                        if checkpoint_payload is None or isinstance(checkpoint_payload, bytes)
                        else checkpoint_payload.encode()
                    ),
                )
            )

        return output

    def save(self) -> list[RunOutputPaths | ItemError]:
        output: list[RunOutputPaths | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                output.append(item)
                continue

            output.append(
                RunOutputPaths(
                    trajectory=item.trajectory.save(ext="xtc"),
                    checkpoint=item.checkpoint.save(ext="bin")
                    if item.checkpoint is not None
                    else None,
                )
            )

        return output


def hyper_run_sumo(
    jobs: list[RunInput],
    *,
    max_inputs: int | None = None,
    nsteps: int | None = None,
    dt_ps: float | None = None,
    temperature_k: float | None = None,
    ensemble: Ensemble | None = None,
    minimize_before_run: bool | None = None,
    solvate_before_run: bool | None = None,
    use_gpu: bool | None = None,
    nthreads: int | None = None,
    timeout_seconds: int | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """Run Hyper molecular dynamics jobs from per-item config/topology/coordinates."""

    job_exprs: list[str] = []
    for job in jobs:
        sim_config_vobj = sim_config_input_to_vobj(job.sim_config_json)
        topology_vobj = topology_input_to_vobj(job.topology)
        coordinates_vobj = trc_object_input_to_vobj(job.coordinates)
        job_exprs.append(
            """(hyper_run_sumo::RunInput {
            sim_config_json = (obj_j "$sim"),
            topology = (obj_j "$topology"),
            coordinates = (obj_j "$coords")
          })"""
            .replace("$sim", sim_config_vobj["path"])
            .replace("$topology", topology_vobj["path"])
            .replace("$coords", coordinates_vobj["path"])
        )

    config_expr = (
        "None"
        if all(
            v is None
            for v in (
                max_inputs,
                nsteps,
                dt_ps,
                temperature_k,
                ensemble,
                minimize_before_run,
                solvate_before_run,
                use_gpu,
                nthreads,
                timeout_seconds,
            )
        )
        else """Some (hyper_run_sumo::HyperRunConfig {
        max_inputs = $max_inputs,
        nsteps = $nsteps,
        dt_ps = $dt_ps,
        temperature_k = $temperature_k,
        ensemble = $ensemble,
        minimize_before_run = $minimize_before_run,
        solvate_before_run = $solvate_before_run,
        use_gpu = $use_gpu,
        nthreads = $nthreads,
        timeout_seconds = $timeout_seconds,
      })"""
    )

    if config_expr != "None":
        config_expr = Template(config_expr).substitute(
            max_inputs=optional_str(max_inputs),
            nsteps=optional_str(nsteps),
            dt_ps=optional_str(dt_ps),
            temperature_k=optional_str(temperature_k),
            ensemble=optional_str(ensemble, prefix="hyper_run_sumo::RunEnsemble::"),
            minimize_before_run=optional_str(minimize_before_run),
            solvate_before_run=optional_str(solvate_before_run),
            use_gpu=optional_str(use_gpu),
            nthreads=optional_str(nthreads),
            timeout_seconds=optional_str(timeout_seconds),
        )

    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  run = λ inputs →
    hyper_run_sumo_s
      ($run_spec)
      ($config)
      inputs
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
