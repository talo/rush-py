"""
Hyper module for the Rush Python client.

This module exposes wrappers for:
- hyper_minimize_sumo
- hyper_run_sumo
- hyper_solvate_sumo
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable, Literal, TypeVar

from gql.transport.exceptions import TransportQueryError

from .._rex import optional_str
from ..convert import from_json, to_dict
from ..mol import TRC
from ..objects import RushObject, fetch_object, upload_object
from ..runs import Run, RunOpts, RunSpec
from ..session import _submit_rex

JsonObjectInput = Path | str | RushObject | dict[str, Any]
TRCInput = TRC | Path | str | RushObject | dict[str, Any]

ErrorStage = Literal["InputDecode", "Execution", "OutputParse"]
ErrorCategory = Literal["InvalidInput", "ToolInput", "OutputFormat"]

TSuccessRef = TypeVar("TSuccessRef")


@dataclass(frozen=True)
class ItemError:
    """Per-item error returned by Hyper batch wrappers."""

    stage: ErrorStage
    category: ErrorCategory
    message: str
    input_index: int

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ItemError":
        if not isinstance(raw, dict):
            raise ValueError(f"Expected ItemError object, got {type(raw).__name__}")
        return cls(
            stage=raw["stage"],
            category=raw["category"],
            message=raw["message"],
            input_index=int(raw["input_index"]),
        )


@dataclass(frozen=True)
class HyperConfig:
    """Config for `hyper_solvate_sumo`."""

    max_inputs: int | None = None
    padding_nm: float | None = None
    seed: int | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class HyperMinimizeConfig:
    """Config for `hyper_minimize_sumo`."""

    max_inputs: int | None = None
    steps: int | None = None
    gtol: float | None = None
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class MinimizeInput:
    """Input item for `hyper_minimize_sumo`."""

    structure: TRCInput
    topology: JsonObjectInput


@dataclass(frozen=True)
class HyperRunConfig:
    """Config for `hyper_run_sumo`."""

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
    """Input item for `hyper_run_sumo`."""

    sim_config: JsonObjectInput
    topology: JsonObjectInput
    coordinates: TRCInput


@dataclass(frozen=True)
class _TRCObjectRef:
    """Reference to one successful TRC output object."""

    output: RushObject


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
class TRCBatchResultRef:
    """Result reference for `hyper_solvate_sumo` and `hyper_minimize_sumo`."""

    items: list[_TRCObjectRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "TRCBatchResultRef":
        parsed = _parse_batch_outputs(raw, _parse_trc_item, "hyper TRC batch")
        return cls(items=parsed)

    def __getitem__(self, index: int) -> _TRCObjectRef | ItemError:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def fetch(self) -> list[TRC | ItemError]:
        out: list[TRC | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                out.append(item)
            else:
                out.append(_fetch_trc(item.output))
        return out

    def save(self) -> list[Path | ItemError]:
        return [
            item if isinstance(item, ItemError) else item.output.save(ext="json")
            for item in self.items
        ]


@dataclass(frozen=True)
class RunResultRef:
    """Result reference for `hyper_run_sumo`."""

    items: list[RunOutputRef | ItemError]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "RunResultRef":
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
            trajectory = fetch_object(item.trajectory.path)
            checkpoint = (
                fetch_object(item.checkpoint.path) if item.checkpoint is not None else None
            )
            out.append(
                RunOutput(
                    trajectory=_ensure_bytes(trajectory),
                    checkpoint=_ensure_bytes(checkpoint) if checkpoint is not None else None,
                )
            )
        return out

    def save(self) -> list[RunOutputPaths | ItemError]:
        out: list[RunOutputPaths | ItemError] = []
        for item in self.items:
            if isinstance(item, ItemError):
                out.append(item)
                continue
            out.append(
                RunOutputPaths(
                    trajectory=item.trajectory.save(ext="xtc"),
                    checkpoint=(
                        item.checkpoint.save(ext="bin")
                        if item.checkpoint is not None
                        else None
                    ),
                )
            )
        return out


def _is_result_type(raw: Any) -> bool:
    return isinstance(raw, dict) and len(raw) == 1 and ("Ok" in raw or "Err" in raw)


def _format_user_error(err: Any) -> str:
    if isinstance(err, str):
        return err
    if isinstance(err, dict) and len(err) == 1:
        key, value = next(iter(err.items()))
        if value is None:
            return str(key)
        if isinstance(value, dict):
            details = ", ".join(f"{k}={v}" for k, v in value.items())
            return f"{key}({details})"
        return f"{key}({value})"
    return json.dumps(err)


def _parse_batch_outputs(
    raw: Any,
    on_success: Callable[[Any], TSuccessRef],
    label: str,
) -> list[TSuccessRef | ItemError]:
    if not isinstance(raw, list) or len(raw) != 1:
        raise ValueError(
            f"{label} should return a single-element list, got {type(raw).__name__}"
        )

    payload = raw[0]
    if _is_result_type(payload):
        if "Err" in payload:
            raise ValueError(f"{label} top-level error: {_format_user_error(payload['Err'])}")
        payload = payload["Ok"]

    items = payload if isinstance(payload, list) else [payload]

    parsed: list[TSuccessRef | ItemError] = []
    for item in items:
        if _is_result_type(item):
            if "Err" in item:
                parsed.append(ItemError.from_raw_output(item["Err"]))
                continue
            item = item["Ok"]
        parsed.append(on_success(item))
    return parsed


def _parse_trc_item(raw: Any) -> _TRCObjectRef:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected TRC output object, got {type(raw).__name__}")
    return _TRCObjectRef(output=RushObject.from_dict(raw))


def _parse_run_item(raw: Any) -> RunOutputRef:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected RunOutput object, got {type(raw).__name__}")

    trajectory = RushObject.from_dict(raw["trajectory"])
    checkpoint_raw = raw.get("checkpoint")
    checkpoint = (
        RushObject.from_dict(checkpoint_raw) if checkpoint_raw is not None else None
    )
    return RunOutputRef(trajectory=trajectory, checkpoint=checkpoint)


def _upload_json_object(input_object: JsonObjectInput) -> RushObject:
    match input_object:
        case RushObject():
            return input_object
        case Path() | str():
            return RushObject.from_dict(upload_object(input_object))
        case dict():
            return RushObject.from_dict(upload_object(input_object))
        case _:
            raise TypeError(
                "Expected Path | str | RushObject | dict input for Hyper JSON object"
            )


def _upload_trc_object(input_object: TRCInput) -> RushObject:
    match input_object:
        case RushObject():
            return input_object
        case TRC():
            trc_dict = to_dict(input_object)
            if not isinstance(trc_dict, dict):
                raise TypeError("Expected single TRC object")
            return RushObject.from_dict(upload_object(trc_dict))
        case Path() | str():
            return RushObject.from_dict(upload_object(input_object))
        case dict():
            return RushObject.from_dict(upload_object(input_object))
        case _:
            raise TypeError(
                "Expected TRC | Path | str | RushObject | dict input for Hyper TRC object"
            )


def _fetch_trc(obj: RushObject) -> TRC:
    raw = fetch_object(obj.path)
    if isinstance(raw, bytes):
        decoded = json.loads(raw.decode())
    elif isinstance(raw, str):
        decoded = json.loads(raw)
    else:
        decoded = raw

    parsed = from_json(decoded)
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise ValueError(f"Expected one TRC object, got {len(parsed)}")
        return parsed[0]
    return parsed


def _ensure_bytes(value: bytes | str) -> bytes:
    return value.encode() if isinstance(value, str) else value


def _to_rex_json_obj(obj: RushObject) -> str:
    return (
        'VirtualObject { path = "'
        + str(obj.path)
        + '", format = ObjectFormat::json, size = 0 }'
    )


def _format_rex_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[\n    " + ",\n    ".join(items) + "\n  ]"


def _to_rex_run_ensemble(value: Literal["Nve", "Nvt", "Npt"] | None) -> str:
    if value is None:
        return "None"
    variants = {
        "Nve": "hyper_run_sumo::RunEnsemble::Nve",
        "Nvt": "hyper_run_sumo::RunEnsemble::Nvt",
        "Npt": "hyper_run_sumo::RunEnsemble::Npt",
    }
    return f"Some {variants[value]}"


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


def hyper_minimize_sumo(
    jobs: list[MinimizeInput],
    config: HyperMinimizeConfig | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> Run[TRCBatchResultRef]:
    """Submit Hyper minimization for one or more structures."""
    job_exprs = []
    for job in jobs:
        structure = _to_rex_json_obj(_upload_trc_object(job.structure))
        topology = _to_rex_json_obj(_upload_json_object(job.topology))
        job_exprs.append(
            "(" +
            "hyper_minimize_sumo::MinimizeInput { "
            f"structure = {structure}, topology = {topology} "
            "}" +
            ")"
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


def hyper_run_sumo(
    jobs: list[RunInput],
    config: HyperRunConfig | None = None,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
) -> Run[RunResultRef]:
    """Submit Hyper molecular dynamics runs for one or more jobs."""
    job_exprs = []
    for job in jobs:
        sim_config = _to_rex_json_obj(_upload_json_object(job.sim_config))
        topology = _to_rex_json_obj(_upload_json_object(job.topology))
        coordinates = _to_rex_json_obj(_upload_trc_object(job.coordinates))
        job_exprs.append(
            "(" +
            "hyper_run_sumo::RunInput { "
            f"sim_config = {sim_config}, topology = {topology}, coordinates = {coordinates} "
            "}" +
            ")"
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


__all__ = [
    "HyperConfig",
    "HyperMinimizeConfig",
    "MinimizeInput",
    "HyperRunConfig",
    "RunInput",
    "ItemError",
    "TRCBatchResultRef",
    "RunOutput",
    "RunOutputRef",
    "RunOutputPaths",
    "RunResultRef",
    "hyper_solvate_sumo",
    "hyper_minimize_sumo",
    "hyper_run_sumo",
]
