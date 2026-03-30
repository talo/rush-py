from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Literal

from gql.transport.exceptions import TransportQueryError

from .._utils import optional_str
from ..client import (
    RunOpts,
    RunSpec,
    RushObject,
    _get_project_id,
    _submit_rex,
    upload_object,
)
from ..run import RushRun
from ._shared import (
    HyperTopologyInput,
    ItemError,
    TRCInput,
    fetch_bytes_object,
    parse_item_results,
    to_hyper_topology_vobj,
    to_trc_vobj,
)

RunEnsemble = Literal["Nve", "Nvt", "Npt"]
StringObjectInput = Path | str | RushObject


@dataclass
class HyperRunConfig:
    max_inputs: int | None = None
    nsteps: int | None = None
    dt_ps: float | None = None
    temperature_k: float | None = None
    ensemble: RunEnsemble | None = None
    minimize_before_run: bool | None = None
    solvate_before_run: bool | None = None
    use_gpu: bool | None = None
    nthreads: int | None = None
    timeout_seconds: int | None = None

    def _to_rex(self) -> str:
        return Template(
            """Some (hyper_run_sumo::HyperRunConfig {
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
        ).substitute(
            max_inputs=optional_str(self.max_inputs),
            nsteps=optional_str(self.nsteps),
            dt_ps=optional_str(self.dt_ps),
            temperature_k=optional_str(self.temperature_k),
            ensemble=optional_str(self.ensemble, "hyper_run_sumo::RunEnsemble::"),
            minimize_before_run=optional_str(self.minimize_before_run),
            solvate_before_run=optional_str(self.solvate_before_run),
            use_gpu=optional_str(self.use_gpu),
            nthreads=optional_str(self.nthreads),
            timeout_seconds=optional_str(self.timeout_seconds),
        )


@dataclass(frozen=True)
class RunInput:
    sim_config_json: StringObjectInput
    topology: HyperTopologyInput
    coordinates: TRCInput


@dataclass(frozen=True)
class RunOutput:
    trajectory: bytes
    checkpoint: bytes | None = None


@dataclass(frozen=True)
class RunOutputPaths:
    trajectory: Path
    checkpoint: Path | None = None


@dataclass(frozen=True)
class _RunOutputRef:
    trajectory: RushObject
    checkpoint: RushObject | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> "_RunOutputRef":
        if not isinstance(raw, dict):
            raise ValueError(
                f"hyper_run_sumo output item should be a dict, got {type(raw).__name__}."
            )

        if "trajectory" not in raw:
            raise ValueError("hyper_run_sumo output item missing required field: trajectory.")

        checkpoint_raw = raw.get("checkpoint")
        return cls(
            trajectory=RushObject.from_dict(raw["trajectory"]),
            checkpoint=(
                RushObject.from_dict(checkpoint_raw)
                if isinstance(checkpoint_raw, dict)
                else None
            ),
        )

    def fetch(self) -> RunOutput:
        return RunOutput(
            trajectory=fetch_bytes_object(self.trajectory),
            checkpoint=(
                fetch_bytes_object(self.checkpoint)
                if self.checkpoint is not None
                else None
            ),
        )

    def save(self) -> RunOutputPaths:
        return RunOutputPaths(
            trajectory=self.trajectory.save(ext="xtc"),
            checkpoint=(
                self.checkpoint.save(ext="cpt")
                if self.checkpoint is not None
                else None
            ),
        )


@dataclass(frozen=True)
class RunResultRef:
    """Per-input result references for `hyper_run_sumo`."""

    _items: list[_RunOutputRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "RunResultRef":
        return cls(
            _items=parse_item_results(
                raw,
                module_name="hyper_run_sumo",
                on_success=_RunOutputRef.from_raw,
            )
        )

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[_RunOutputRef | ItemError]:
        return iter(self._items)

    def __getitem__(self, index: int) -> _RunOutputRef | ItemError:
        return self._items[index]

    def fetch(self) -> list[RunOutput | ItemError]:
        return [
            item.fetch() if isinstance(item, _RunOutputRef) else item
            for item in self._items
        ]

    def save(self) -> list[RunOutputPaths | ItemError]:
        return [
            item.save() if isinstance(item, _RunOutputRef) else item
            for item in self._items
        ]


def _to_string_object_vobj(value: StringObjectInput) -> dict[str, Any]:
    if isinstance(value, RushObject):
        return value.to_dict()
    return upload_object(value)


def hyper_run_sumo(
    jobs: Sequence[RunInput],
    config: HyperRunConfig | None = None,
    run_spec: RunSpec = RunSpec(storage=4096),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[RunResultRef]:
    """Run Hyper MD simulations from JSON config + topology + coordinates inputs."""

    job_exprs: list[str] = []
    for job in jobs:
        sim_config_vobj = _to_string_object_vobj(job.sim_config_json)
        topology_vobj = to_hyper_topology_vobj(job.topology)
        coordinates_vobj = to_trc_vobj(job.coordinates)

        job_exprs.append(
            Template(
                """(hyper_run_sumo::RunInput {
          sim_config_json = (obj_j \"$sim_config_json_path\"),
          topology = (obj_j \"$topology_path\"),
          coordinates = (obj_j \"$coordinates_path\"),
        })"""
            ).substitute(
                sim_config_json_path=sim_config_vobj["path"],
                topology_path=topology_vobj["path"],
                coordinates_path=coordinates_vobj["path"],
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
  hyper_run_sumo_s
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
            RunResultRef,
        )
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
