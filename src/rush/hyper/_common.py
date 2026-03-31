from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeGuard, TypeVar

from rush import TRC, from_json, to_dict

from .._utils import float_to_str, optional_str
from ..client import RushObject, fetch_object, upload_object

T = TypeVar("T")


JsonInput = Path | str | RushObject | dict[str, Any]
TRCInput = TRC | JsonInput


@dataclass(frozen=True)
class ItemError:
    """Per-input failure returned by Hyper batch entrypoints."""

    stage: str
    category: str
    message: str
    input_index: int

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ItemError":
        if not isinstance(raw, dict):
            raise ValueError(f"ItemError should be a dict, got {type(raw).__name__}.")

        required = {"stage", "category", "message", "input_index"}
        if not required.issubset(raw):
            raise ValueError(
                f"ItemError missing keys: {sorted(required - set(raw.keys()))}."
            )

        return cls(
            stage=str(raw["stage"]),
            category=str(raw["category"]),
            message=str(raw["message"]),
            input_index=int(raw["input_index"]),
        )


@dataclass(frozen=True)
class HyperRunOutputRef:
    """Reference to one successful Hyper run output."""

    trajectory: RushObject
    checkpoint: RushObject | None


@dataclass(frozen=True)
class HyperRunOutput:
    """Fetched bytes for one successful Hyper run output."""

    trajectory: bytes
    checkpoint: bytes | None


@dataclass(frozen=True)
class HyperRunOutputPaths:
    """Workspace paths for one saved Hyper run output."""

    trajectory: Path
    checkpoint: Path | None


def is_result_type(value: Any) -> TypeGuard[dict[str, Any]]:
    return (
        isinstance(value, dict)
        and len(value) == 1
        and ("Ok" in value or "Err" in value)
    )


def parse_fallible_items(raw: Any, parse_ok: Callable[[Any], T]) -> list[T | ItemError]:
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of fallible item outputs, got {type(raw).__name__}.")

    parsed: list[T | ItemError] = []
    for index, item in enumerate(raw):
        if not is_result_type(item):
            raise ValueError(
                "Expected each item to be a fallible {'Ok': ...} or {'Err': ...} record; "
                f"item {index} is {type(item).__name__}."
            )

        if "Ok" in item:
            parsed.append(parse_ok(item["Ok"]))
        else:
            parsed.append(ItemError.from_raw_output(item["Err"]))

    return parsed


def to_json_vobj(item: JsonInput) -> dict[str, Any]:
    if isinstance(item, RushObject):
        return item.to_dict()
    return upload_object(item)


def to_trc_vobj(item: TRCInput) -> dict[str, Any]:
    if isinstance(item, TRC):
        serialized = to_dict(item)
        if not isinstance(serialized, dict):
            raise ValueError("Expected a single TRC object for upload.")
        return upload_object(serialized)
    return to_json_vobj(item)


def fetch_json_dict(obj: RushObject) -> dict[str, Any]:
    raw = fetch_object(obj.path)
    text = raw.decode() if isinstance(raw, bytes) else raw
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected JSON object in {obj.path}, got {type(parsed).__name__}."
        )
    return parsed


def fetch_trc(obj: RushObject) -> TRC:
    parsed = from_json(fetch_json_dict(obj))
    if not isinstance(parsed, TRC):
        raise ValueError(
            f"Expected a single TRC object in {obj.path}, got {type(parsed).__name__}."
        )
    return parsed


def fetch_bytes(obj: RushObject) -> bytes:
    raw = fetch_object(obj.path)
    if isinstance(raw, bytes):
        return raw
    return raw.encode()


def as_hyper_solvate_config(
    *,
    max_inputs: int | None,
    padding_nm: float | None,
    seed: int | None,
    timeout_seconds: int | None,
) -> str:
    if (
        max_inputs is None
        and padding_nm is None
        and seed is None
        and timeout_seconds is None
    ):
        return "(None)"

    return (
        "(Some (hyper_solvate_sumo::HyperConfig {"
        f" max_inputs = {optional_str(max_inputs)},"
        f" padding_nm = {optional_str(padding_nm)},"
        f" seed = {optional_str(seed)},"
        f" timeout_seconds = {optional_str(timeout_seconds)}"
        " }))"
    )


def as_hyper_minimize_config(
    *,
    max_inputs: int | None,
    steps: int | None,
    gtol: float | None,
    timeout_seconds: int | None,
) -> str:
    if max_inputs is None and steps is None and gtol is None and timeout_seconds is None:
        return "(None)"

    return (
        "(Some (hyper_minimize_sumo::HyperMinimizeConfig {"
        f" max_inputs = {optional_str(max_inputs)},"
        f" steps = {optional_str(steps)},"
        f" gtol = {optional_str(gtol)},"
        f" timeout_seconds = {optional_str(timeout_seconds)}"
        " }))"
    )


def as_hyper_run_config(
    *,
    max_inputs: int | None,
    nsteps: int | None,
    dt_ps: float | None,
    temperature_k: float | None,
    ensemble: str | None,
    minimize_before_run: bool | None,
    solvate_before_run: bool | None,
    use_gpu: bool | None,
    nthreads: int | None,
    timeout_seconds: int | None,
) -> str:
    if (
        max_inputs is None
        and nsteps is None
        and dt_ps is None
        and temperature_k is None
        and ensemble is None
        and minimize_before_run is None
        and solvate_before_run is None
        and use_gpu is None
        and nthreads is None
        and timeout_seconds is None
    ):
        return "(None)"

    ensemble_value = (
        f"Some hyper_run_sumo::RunEnsemble::{ensemble}"
        if ensemble is not None
        else "None"
    )

    dt_ps_value = f"Some {float_to_str(dt_ps)}" if dt_ps is not None else "None"
    temperature_k_value = (
        f"Some {float_to_str(temperature_k)}" if temperature_k is not None else "None"
    )

    return (
        "(Some (hyper_run_sumo::HyperRunConfig {"
        f" max_inputs = {optional_str(max_inputs)},"
        f" nsteps = {optional_str(nsteps)},"
        f" dt_ps = {dt_ps_value},"
        f" temperature_k = {temperature_k_value},"
        f" ensemble = {ensemble_value},"
        f" minimize_before_run = {optional_str(minimize_before_run)},"
        f" solvate_before_run = {optional_str(solvate_before_run)},"
        f" use_gpu = {optional_str(use_gpu)},"
        f" nthreads = {optional_str(nthreads)},"
        f" timeout_seconds = {optional_str(timeout_seconds)}"
        " }))"
    )
