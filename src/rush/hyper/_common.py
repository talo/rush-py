"""Shared parsing and upload helpers for Hyper entrypoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

from ..convert import from_json, to_dict
from ..mol import TRC
from ..objects import RushObject, upload_object

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


def _upload_json_object(input_object: JsonObjectInput) -> RushObject:
    match input_object:
        case RushObject():
            return input_object
        case Path() | str() | dict():
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
        case Path() | str() | dict():
            return RushObject.from_dict(upload_object(input_object))
        case _:
            raise TypeError(
                "Expected TRC | Path | str | RushObject | dict input for Hyper TRC object"
            )


def _fetch_trc(obj: RushObject) -> TRC:
    parsed = from_json(obj.fetch_json())
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise ValueError(f"Expected one TRC object, got {len(parsed)}")
        item = parsed[0]
        if not isinstance(item, TRC):
            raise TypeError("Expected TRC item in parsed list")
        return item
    if not isinstance(parsed, TRC):
        raise TypeError(f"Expected TRC output, got {type(parsed).__name__}")
    return parsed


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
