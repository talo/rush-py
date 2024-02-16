#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


T = TypeVar("T")


if sys.version_info >= (3, 12):
    from typing import Optional as _Optional  # noqa: F401

    # For type styling in docs
    exec("type Optional[T] = _Optional[T]")
    exec("type Conformer = dict[str, Any]")
    exec("type Record = dict[str, Any]")
    exec("type EnumValue = str")
else:
    from typing import Optional

    try:
        from typing import TypeAliasType
    except ImportError:
        from typing_extensions import TypeAliasType
    Conformer = TypeAliasType("Conformer", dict[str, Any])
    Record = TypeAliasType("Record", dict[str, Any])
    EnumValue = TypeAliasType("EnumValue", str)

U = TypeVar("U", bytes, Conformer, Record, list[Any], float)


class _RushObject(Generic[U]):
    object: U | None = None


if sys.version_info >= (3, 12):
    exec("type RushObject[U] = _RushObject[Any]")
else:
    RushObject = TypeAliasType("RushObject", _RushObject[U], type_params=(U,))


SCALARS = Literal[
    "bool", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "string", "bytes", "Conformer"
]

SCALAR_STRS: list[SCALARS] = [
    "bool",
    "u8",
    "u16",
    "u32",
    "u64",
    "i8",
    "i16",
    "i32",
    "i64",
    "f32",
    "f64",
    "string",
    "bytes",
    "Conformer",
]


scalar_types_mapping: dict[str, type[Any]] = {
    "bool": bool,
    "u8": int,
    "u16": int,
    "u32": int,
    "u64": int,
    "i8": int,
    "i16": int,
    "i32": int,
    "i64": int,
    "f32": float,
    "f64": float,
    "string": str,
    "bytes": bytes,
    "Conformer": Conformer,
}


KINDS = Literal["array", "optional", "enum", "record", "object", "tuple"]


@dataclass
class SimpleType:
    k: KINDS | None
    t: SCALARS | "SimpleType"


@dataclass
class Tagged:
    n: str
    t: SimpleType


class RushType(Generic[T]):
    def __init__(
        self, type: "dict[str, RushType[T]] | list[RushType[T]] | RushType[T]", kind: KINDS | None = None
    ):
        self.k = kind
        self.t = type

    def to_python_type(self) -> type[Any]:
        raise Exception("Invalid type")

    def matches(self, _: Any) -> tuple[bool, str | None]:
        """
        Compare a Type to a python object. Returns true if the object matches the type.
        """
        raise Exception("Invalid type")


class EnumKind(Generic[T], RushType[T]):
    def __init__(self, enum: list[str | dict[str, RushType[T]]]):
        self.k = "enum"
        self.t = enum
        self.literals = [x for x in enum if isinstance(x, str)]
        self.tags = [x for x in enum if isinstance(x, dict)]

    def to_python_type(self) -> type[EnumValue]:
        return EnumValue

    def matches(self, other: str | dict[str, RushType[T]] | Any) -> tuple[bool, str | None]:
        if isinstance(other, dict) and len(other) == 1:
            other_key = list(other.keys())[0]
            for t in self.tags:
                if list(t.keys())[0] == other_key:
                    return t[other_key].matches(other[other_key])
            return (False, f"Unknown enum variant {other_key}")
        if isinstance(other, str):
            # FIXME: make case sensitive once enum serialization case bug is fixed
            if any([x.lower() == other.lower() for x in self.literals]):
                return (True, None)

        return (False, f"Unknown enum variant {other}")


class RecordKind(Generic[T], RushType[T]):
    def __init__(self, record: dict[str, RushType[T]]):
        self.k = "record"
        self.t = record

    def to_python_type(self) -> type[Record]:
        return Record

    def matches(self, other: dict[str, Any] | Any) -> tuple[bool, str | None]:
        if not isinstance(other, dict):
            return (False, f"Expected dict, got {type(other)}")
        for k, v in self.t.items():
            if v.k == "optional" and k not in other:
                return (True, None)
            if k not in other and v.k != "optional":
                return (False, f"Expected key {k} in dict")
            ok, reason = v.matches(other[k])
            if not ok:
                return (False, reason)
        return (True, None)


class ArrayKind(Generic[T], RushType[T]):
    def __init__(self, array: RushType[T]):
        self.k = "array"
        self.t = array

    def to_python_type(self) -> type[list[Any]]:
        return list[self.t.to_python_type()]

    def matches(self, other: list[T] | Any) -> tuple[bool, str | None]:
        if not (isinstance(other, list) or isinstance(other, tuple)):
            return (False, f"Expected list or tuple, got {type(other)}")
        for x in other:
            ok, reason = self.t.matches(x)
            if not ok:
                return (False, reason)
        return (True, None)


class TupleKind(Generic[T], RushType[T]):
    def __init__(self, tuple: list[RushType[T]]):
        self.k = "tuple"
        self.t = tuple

    def to_python_type(self) -> type[tuple[Any]]:
        if sys.version_info >= (3, 11):
            return exec("tuple[*(t.to_python_type() for t in self.t)]")  # type: ignore
        else:
            return tuple[tuple(t.to_python_type() for t in self.t)]  # type: ignore

    def matches(self, other: tuple[T] | list[T] | Any) -> tuple[bool, str | None]:
        if not (isinstance(other, list) or isinstance(other, tuple)):
            return (False, f"Expected list or tuple, got {type(other)}")
        if len(other) != len(self.t):
            return (False, f"Expected list of length {len(self.t)}, got {len(other)}")
        for x, y in zip(self.t, other):
            ok, reason = x.matches(y)
            if not ok:
                return (False, reason)
        return (True, None)


class OptionalKind(Generic[T], RushType[T]):
    def __init__(self, optional: RushType[T]):
        self.k = "optional"
        self.t = optional

    def to_python_type(self) -> type[Any] | None:
        return Optional[self.t.to_python_type()]

    def matches(self, other: T | None | Any) -> tuple[bool, str | None]:
        if other is None:
            return (True, None)
        else:
            return self.t.matches(other)


class ObjectKind(Generic[T], RushType[T]):
    def __init__(self, object: RushType[T]):
        self.k = "object"
        self.t = object

    def to_python_type(self) -> type[RushObject[Any]]:
        return RushObject[self.t.to_python_type()]

    def matches(self, other: Path | Any) -> tuple[bool, str | None]:
        if isinstance(other, _RushObject) or isinstance(other, Path):
            return (True, None)
        else:
            return (False, f"Expected Path, got {type(other)}")


class ScalarType(Generic[T], RushType[T]):
    def __init__(self, scalar: SCALARS | str):
        self.k = None
        self.t = scalar
        self.py_type = scalar_types_mapping.get(scalar)
        self.literal = scalar if not self.py_type else None

    def to_python_type(self) -> type[Any] | None:
        return self.py_type

    def matches(self, other: T) -> tuple[bool, str | None]:
        if self.literal:
            if other == self.literal:
                return (True, None)
            else:
                return (False, f"Expected {self.literal}, got {other}")
        elif not self.py_type:
            return (True, None)
        elif isinstance(other, self.py_type):
            return (True, None)
        else:
            return (False, f"Expected {self.py_type}, got {other}")


def type_from_typedef(res: Any) -> RushType[Any]:
    if isinstance(res, dict):
        if res.get("k"):
            if res["k"] == "enum":
                literals = [x for x in res["t"] if isinstance(x, str)]
                tags = [
                    {list(x.keys())[0]: type_from_typedef(list(x.values())[0])}
                    for x in res["t"]
                    if isinstance(x, dict)
                ]
                return EnumKind(literals + tags)
            elif res["k"] == "record":
                return RecordKind({k: type_from_typedef(v) for k, v in res["t"].items()})
            elif res["k"] == "array":
                return ArrayKind(type_from_typedef(res["t"]))
            elif res["k"] == "tuple":
                return TupleKind([type_from_typedef(x) for x in res["t"]])
            elif res["k"] == "optional":
                return OptionalKind(type_from_typedef(res["t"]))
            elif res["k"] == "object":
                return ObjectKind(type_from_typedef(res["t"]))
            else:
                raise Exception(f"Unknown kind {res['k']}")
        else:
            if res.get("t"):
                return type_from_typedef(res["t"])
            else:
                raise Exception(f"Invalid typedef {res}")
    elif isinstance(res, list):
        return TupleKind([type_from_typedef(x) for x in res])
    elif isinstance(res, str):  # type: ignore
        return ScalarType(res)
    else:
        print("Bad type!")
        return RushType(res)


def build_typechecker(
    *types: Unpack[tuple[RushType[Any], ...]],
):
    def built(*args: Any):
        for t, a in zip(types, args):
            match = t.matches(a)
            if not match[0]:
                # if args are references, let the api check them
                if "__dict__" in a.__dir__() and a.__dict__.get("id") or isinstance(a, UUID):
                    return True
                else:
                    raise Exception(f"Typecheck failed: {match[1]}")

    return built
