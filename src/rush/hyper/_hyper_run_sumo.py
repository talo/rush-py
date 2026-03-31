"""Hyper molecular dynamics wrapper for the Rush Python client."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Literal, Self

from gql.transport.exceptions import TransportQueryError

from .._rex import optional_str
from ..objects import RushObject
from ..runs import Run, RunOpts, RunSpec
from ..session import _submit_rex
from ._common import (
    ItemError,
    JsonObjectInput,
    TRCInput,
    _format_rex_list,
    _parse_batch_outputs,
    _to_rex_json_obj,
    _upload_json_object,
    _upload_trc_object,
)


@dataclass(frozen=True)
class HyperRunConfig:
    """Config for :func:`hyper_run_sumo`."""

    max_inputs: int | None = None
    nsteps: int | None = None
    dt_ps: float | None = None
    temperature_k: float | None = None
    ensemble: Literal["Nve", "Nvt", "Npt"] | None = None
    minimize_before_run: bool | None = None
    solvate_before_run: bool | None = None
    use_gpu: bool | None = None
    nthreads: int | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class RunInput:
    """Input item for :func:`hyper_run_sumo`."""

    sim_config: JsonObjectInput
    topology: JsonObjectInput
    coordinates: TRCInput


@dataclass(frozen=True)
class RunOutputRef:
    """Reference to one successful run output."""

    trajectory: RushObject
    checkpoint: RushObject | None


@dataclass(frozen=True)
class RunOutput:
    """Fetched bytes for one successful run output."""

    trajectory: bytes
    checkpoint: bytes | None


@dataclass(frozen=True)
class RunOutputPaths:
    """Workspace paths for one saved run output."""

    trajectory: Path
    checkpoint: Path | None


@dataclass(frozen=True)
class RunResultRef:
    """Result reference for :func:`hyper_run_sumo`."""

    items: list[RunOutputRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> Self:
        parsed = _parse_batch_outputs(raw, _parse_run_item, "hyper run batch")
        return cls(items=parsed)

    def __getitem__(self, index: int) -> RunOutputRef | ItemError:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def fetch(self) -> list[RunOutput | ItemError]:
        out: list[RunOutput | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                out.append(item)
                continue
            trajectory = _fetch_run_artifact_bytes(item.trajectory, "trajectory")
            checkpoint = (
                _fetch_run_artifact_bytes(item.checkpoint, "checkpoint")
                if item.checkpoint is not None
                else None
            )
            out.append(RunOutput(trajectory=trajectory, checkpoint=checkpoint))
        return out

    def save(self) -> list[RunOutputPaths | ItemError]:
        out: list[RunOutputPaths | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                out.append(item)
                continue
            out.append(
                RunOutputPaths(
                    trajectory=_save_run_artifact(
                        item.trajectory,
                        ext="xtc",
                        label="trajectory",
                    ),
                    checkpoint=(
                        _save_run_artifact(
                            item.checkpoint,
                            ext="bin",
                            label="checkpoint",
                        )
                        if item.checkpoint is not None
                        else None
                    ),
                )
            )
        return out


def _fetch_run_artifact_bytes(obj: RushObject, label: str) -> bytes:
    if obj.format.lower() == "bin":
        return obj.fetch_bytes()

    if obj.format.lower() != "json":
        raise TypeError(
            f"hyper_run_sumo {label} object has unsupported format {obj.format!r}"
        )

    payload = obj.fetch_list()
    try:
        return bytes(payload)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"hyper_run_sumo {label} JSON output must be a list of byte values"
        ) from exc


def _save_run_artifact(obj: RushObject, ext: str, label: str) -> Path:
    out_path = obj.save(ext=ext)
    if obj.format.lower() == "json":
        payload = _fetch_run_artifact_bytes(obj, label)
        with out_path.open("wb") as out_file:
            out_file.write(payload)
    return out_path


def _parse_run_item(raw: Any) -> RunOutputRef:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected RunOutput object, got {type(raw).__name__}")

    trajectory = RushObject.from_dict(raw["trajectory"])
    checkpoint_raw = raw.get("checkpoint")
    checkpoint = RushObject.from_dict(checkpoint_raw) if checkpoint_raw is not None else None
    return RunOutputRef(trajectory=trajectory, checkpoint=checkpoint)


def _to_rex_run_ensemble(value: Literal["Nve", "Nvt", "Npt"] | None) -> str:
    if value is None:
        return "None"
    variants = {
        "Nve": "hyper_run_sumo::RunEnsemble::Nve",
        "Nvt": "hyper_run_sumo::RunEnsemble::Nvt",
        "Npt": "hyper_run_sumo::RunEnsemble::Npt",
    }
    return f"Some {variants[value]}"


def _to_rex_run_config(config: HyperRunConfig | None) -> str:
    if config is None:
        return "None"
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
        max_inputs=optional_str(config.max_inputs),
        nsteps=optional_str(config.nsteps),
        dt_ps=optional_str(config.dt_ps),
        temperature_k=optional_str(config.temperature_k),
        ensemble=_to_rex_run_ensemble(config.ensemble),
        minimize_before_run=optional_str(config.minimize_before_run),
        solvate_before_run=optional_str(config.solvate_before_run),
        use_gpu=optional_str(config.use_gpu),
        nthreads=optional_str(config.nthreads),
        timeout_seconds=optional_str(config.timeout_seconds),
    )


def hyper_run_sumo(
    jobs: list[RunInput],
    config: HyperRunConfig | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> Run[RunResultRef]:
    """Submit Hyper molecular dynamics runs for one or more jobs."""
    job_exprs: list[str] = []
    for job in jobs:
        sim_config = _to_rex_json_obj(_upload_json_object(job.sim_config))
        topology = _to_rex_json_obj(_upload_json_object(job.topology))
        coordinates = _to_rex_json_obj(_upload_trc_object(job.coordinates))
        job_exprs.append(
            "("
            + "hyper_run_sumo::RunInput { "
            + (
                f"sim_config = {sim_config}, topology = {topology}, "
                f"coordinates = {coordinates} "
            )
            + "}"
            + ")"
        )

    rex = Template(
        """hyper_run_sumo_s
  ($run_spec)
  ($config)
  $jobs"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        config=_to_rex_run_config(config),
        jobs=_format_rex_list(job_exprs),
    )

    try:
        return Run(_submit_rex(rex, run_opts), RunResultRef)
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
