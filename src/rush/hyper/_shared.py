from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, TypeGuard, TypeVar, cast

from rush import TRC, TRCRef

from ..client import RushObject, fetch_object, upload_object
from ..convert import from_json, to_dict

TRCInput = TRC | TRCRef | Path | str | RushObject
HyperTopologyInput = dict[str, Any] | Path | str | RushObject

ErrorStage = Literal["InputDecode", "Execution", "OutputParse"]
ErrorCategory = Literal["InvalidInput", "ToolInput", "OutputFormat"]


@dataclass(frozen=True)
class ItemError:
    """Per-item error returned by hyper_*_sumo entrypoints."""

    stage: ErrorStage
    category: ErrorCategory
    message: str
    input_index: int


def is_result_type(value: Any) -> TypeGuard[dict[str, Any]]:
    return (
        isinstance(value, dict)
        and len(value) == 1
        and ("Ok" in value or "Err" in value)
    )


T = TypeVar("T")


def parse_item_results(
    raw: Any,
    *,
    module_name: str,
    on_success: Callable[[Any], T],
) -> list[T | ItemError]:
    """Parse per-item `Ok/Err` outputs from Hyper modules."""

    if not isinstance(raw, list):
        raise ValueError(
            f"{module_name} should return a list of per-item results, got {type(raw).__name__}."
        )

    parsed: list[T | ItemError] = []
    for item in raw:
        if is_result_type(item):
            if "Ok" in item:
                parsed.append(on_success(item["Ok"]))
            else:
                parsed.append(_parse_item_error(item["Err"], module_name=module_name))
            continue

        # Defensive fallback if collect_run changes unwrapping behavior.
        parsed.append(on_success(item))

    return parsed


def _parse_item_error(raw: Any, *, module_name: str) -> ItemError:
    if not isinstance(raw, dict):
        raise ValueError(
            f"{module_name} item error should be a dict, got {type(raw).__name__}."
        )

    required = ("stage", "category", "message", "input_index")
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(
            f"{module_name} item error missing required field(s): {', '.join(missing)}."
        )

    return ItemError(
        stage=raw["stage"],
        category=raw["category"],
        message=str(raw["message"]),
        input_index=int(raw["input_index"]),
    )


def to_trc_vobj(value: TRCInput) -> dict[str, Any]:
    """Convert a TRC-like value into a Rush object descriptor."""

    if isinstance(value, TRC):
        payload = to_dict(value)
        if isinstance(payload, list):
            raise ValueError("Expected a single TRC object, got a list of TRCs.")
        return upload_object(cast(dict[str, Any], payload))

    if isinstance(value, TRCRef):
        # TRCRef is stored as a triplet; hyper modules expect a single Object<TRC>.
        payload = to_dict(value.fetch())
        if isinstance(payload, list):
            raise ValueError("Expected a single TRC object, got a list of TRCs.")
        return upload_object(cast(dict[str, Any], payload))
    if isinstance(value, RushObject):
        return value.to_dict()

    if isinstance(value, (Path, str)):
        return upload_object(value)

    raise TypeError(f"Cannot convert {type(value)} to Object<TRC>.")


def to_hyper_topology_vobj(value: HyperTopologyInput) -> dict[str, Any]:
    """Convert HyperTopology input into a Rush object descriptor."""

    if isinstance(value, RushObject):
        return value.to_dict()

    if isinstance(value, dict):
        return upload_object(value)

    if isinstance(value, (Path, str)):
        return upload_object(value)

    raise TypeError(f"Cannot convert {type(value)} to Object<HyperTopology>.")


def fetch_trc_object(obj: RushObject) -> TRC:
    """Download an Object<TRC> and parse it into a TRC dataclass."""

    payload = fetch_object(obj.path)
    if isinstance(payload, bytes):
        payload = json.loads(payload.decode())
    elif isinstance(payload, str):
        payload = json.loads(payload)

    return from_json(payload)


def fetch_bytes_object(obj: RushObject) -> bytes:
    payload = fetch_object(obj.path)
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode()
    return json.dumps(payload).encode()
