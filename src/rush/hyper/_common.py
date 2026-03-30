from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, TypeGuard, TypeVar

from ..mol import TRC

from .._trc import TRCRef
from ..client import RushObject, upload_object
from ..convert import to_dict as trc_to_dict


@dataclass(frozen=True)
class ItemError:
    """Per-item Hyper failure payload returned by fallible batch outputs."""

    stage: str
    category: str
    message: str
    input_index: int

    @classmethod
    def from_raw(cls, raw: Any, *, default_index: int) -> "ItemError":
        if isinstance(raw, dict):
            stage = raw.get("stage")
            category = raw.get("category")
            message = raw.get("message")
            input_index = raw.get("input_index")
            if (
                isinstance(stage, str)
                and isinstance(category, str)
                and isinstance(message, str)
                and isinstance(input_index, int)
            ):
                return cls(
                    stage=stage,
                    category=category,
                    message=message,
                    input_index=input_index,
                )
        if isinstance(raw, str):
            return cls(
                stage="Execution",
                category="ToolInput",
                message=raw,
                input_index=default_index,
            )
        raise ValueError(
            f"Hyper item error payload has unsupported shape: {type(raw).__name__}"
        )


T = TypeVar("T")


def _is_result_wrapper(value: Any) -> TypeGuard[dict[str, Any]]:
    return (
        isinstance(value, dict)
        and len(value) == 1
        and ("Ok" in value or "Err" in value)
    )


def parse_fallible_items(raw: Any, *, parse_ok: Callable[[Any], T]) -> list[T | ItemError]:
    if not isinstance(raw, list):
        raise ValueError(f"Expected list output, got {type(raw).__name__}")

    parsed: list[T | ItemError] = []
    for index, item in enumerate(raw):
        if _is_result_wrapper(item):
            if "Ok" in item:
                parsed.append(parse_ok(item["Ok"]))
            else:
                parsed.append(ItemError.from_raw(item["Err"], default_index=index))
            continue

        # Some runtimes already unwrap per-item Result values.
        parsed.append(parse_ok(item))

    return parsed


def _trc_to_uploadable_dict(trc: TRC) -> dict[str, Any]:
    payload = trc_to_dict(trc)
    if isinstance(payload, list):
        raise TypeError("Expected a single TRC payload, got a TRC list payload.")
    return payload


def trc_object_input_to_vobj(
    value: TRC | TRCRef | Path | str | RushObject,
) -> dict[str, Any]:
    if isinstance(value, RushObject):
        return value.to_dict()

    if isinstance(value, TRC):
        return upload_object(_trc_to_uploadable_dict(value))

    if isinstance(value, TRCRef):
        return upload_object(_trc_to_uploadable_dict(value.fetch()))

    return upload_object(Path(value))

def topology_input_to_vobj(value: Path | str | RushObject | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, RushObject):
        return value.to_dict()

    if isinstance(value, dict):
        return upload_object(value)

    return upload_object(Path(value))


def sim_config_input_to_vobj(value: Path | str | RushObject) -> dict[str, Any]:
    if isinstance(value, RushObject):
        return value.to_dict()

    if isinstance(value, Path):
        return upload_object(value)

    maybe_path = Path(value)
    if maybe_path.exists():
        return upload_object(maybe_path)

    # Allow passing the config JSON string directly.
    temp_file = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        temp_file.write(value)
        temp_file.close()
        return upload_object(Path(temp_file.name))
    finally:
        Path(temp_file.name).unlink(missing_ok=True)
