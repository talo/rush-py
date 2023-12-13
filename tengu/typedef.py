#!/usr/bin/env python3

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Union
import typing
from uuid import UUID

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

T = typing.TypeVar("T")
U = typing.TypeVar("U")
V = typing.TypeVar("V")


scalar_types_mapping = {
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
    "Conformer": dict,
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


class Type(typing.Generic[T]):
    def __init__(
        self, type: Union[dict[str, "Type[T]"], list["Type[T]"], "Type[T]"], kind: KINDS | None = None
    ):
        self.k: KINDS | None = kind
        self.t = type

    def to_python_type(self):
        raise Exception("Invalid type")

    def matches(self, _: Any) -> tuple[bool, str | None]:
        """
        Compare a Type to a python object. Returns true if the object matches the type.
        """
        raise Exception("Invalid type")


class EnumKind(typing.Generic[T], Type[T]):
    def __init__(self, enum: list[str | dict[str, Type[T]]]):
        self.k = "enum"
        self.t = enum
        self.literals = [x for x in enum if isinstance(x, str)]
        self.tags = [x for x in enum if isinstance(x, dict)]

    def to_python_type(self):
        return list[str | T]

    def matches(self, other: str | dict[str, Type[T]] | Any) -> tuple[bool, str | None]:
        if isinstance(other, dict) and len(other) == 1:
            other_key = list(other.keys())[0]
            for t in self.tags:
                if list(t.keys())[0] == other_key:
                    return t[other_key].matches(other[other_key])
            return (False, f"Unknown enum variant {other_key}")
        if isinstance(other, str):
            if any([x == other for x in self.literals]):
                return (True, None)

        return (False, f"Unknown enum variant {other}")


class RecordKind(typing.Generic[T], Type[T]):
    def __init__(self, record: dict[str, Type[T]]):
        self.k = "record"
        self.t = record

    def to_python_type(self):
        return dict[str, T]

    def matches(self, other: dict[str, Any] | Any) -> tuple[bool, str | None]:
        if not isinstance(other, dict):
            return (False, f"Expected dict, got {type(other)}")
        for k, v in self.t.items():
            if k not in other:
                return (False, f"Expected key {k} in dict")
            ok, reason = v.matches(other[k])
            if not ok:
                return (False, reason)
        return (True, None)


class ArrayKind(typing.Generic[T], Type[T]):
    def __init__(self, array: Type[T]):
        self.k = "array"
        self.t = array

    def to_python_type(self):
        return list[T]

    def matches(self, other: list[T] | Any) -> tuple[bool, str | None]:
        if not (isinstance(other, list) or isinstance(other, tuple)):
            return (False, f"Expected list or tuple, got {type(other)}")
        for x in other:
            ok, reason = self.t.matches(x)
            if not ok:
                return (False, reason)
        return (True, None)


class TupleKind(typing.Generic[T], Type[T]):
    def __init__(self, tuple: list[Type[T]]):
        self.k = "tuple"
        self.t = tuple

    def to_python_type(self):
        return tuple[T]

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


class OptionalKind(typing.Generic[T], Type[T]):
    def __init__(self, optional: Type[T]):
        self.k = "optional"
        self.t = optional

    def to_python_type(self):
        return T | None

    def matches(self, other: T | None | Any) -> tuple[bool, str | None]:
        if other is None:
            return (True, None)
        else:
            return self.t.matches(other)


class ObjectKind(typing.Generic[T], Type[T]):
    def __init__(self, object: Type[T]):
        self.k = "object"
        self.t = object

    def to_python_type(self):
        return Path

    def matches(self, other: Path | Any) -> tuple[bool, str | None]:
        if isinstance(other, Path):
            return (True, None)
        else:
            return (False, f"Expected Path, got {type(other)}")


class ScalarType(typing.Generic[T], Type[T]):
    def __init__(self, scalar: SCALARS | "str"):
        self.k = None
        self.t = self  # scalar
        self.py_type = scalar_types_mapping.get(scalar)
        self.literal = scalar if not self.py_type else None

    def to_python_type(self):
        return self.py_type

    def matches(self, other: T) -> tuple[bool, str | None]:
        if self.literal:
            if other == self.literal:
                return (True, None)
            else:
                return (False, f"Expected {self.literal}, got {other}")
        if isinstance(other, self.py_type):
            return (True, None)
        else:
            return (False, f"Expected {self.py_type}, got {other}")


def type_from_typedef(res: Any) -> Type[Any]:
    if isinstance(res, dict):
        if res.get("k"):
            if res["k"] == "enum":
                literals = [x for x in res["t"] if isinstance(x, str)]
                tags = [
                    {list(x.keys())[0]: type_from_typedef(list(x.values())[0])}
                    for x in res["t"]
                    if isinstance(x, dict)
                ]

                x = None
                for i in literals:
                    if x:
                        x = Union[Literal[i], x]
                    else:
                        x = Literal[i]

                for i in tags:
                    if x:
                        x = Union[type(i), x]
                    else:
                        x = type(i)

                return EnumKind[x](literals + tags)
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
    else:
        return ScalarType(res) if isinstance(res, str) else Type(res)


sample_typedef_json = '{"k":"enum","t":["Foo",{"Bar":["i32","f32","string",{"k":"tuple","t":["i32","f32","string"]}]},{"Baz":{"f":"f32","i":"i32","s":"string","t":{"k":"tuple","t":["i32","f32","string"]}}}]}'


def build_typechecker(
    *types: Type[Any],
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


def test():
    f = build_typechecker(
        ObjectKind(ScalarType[Literal["bytes"]]("bytes")),
        RecordKind({"a": ScalarType[Literal["i32"]]("i32")}),
        RecordKind({"b": ScalarType("i32")}),
    )
    f(Path("."), {"a": 1}, {"b": 2})

    f2 = build_typechecker(type_from_typedef(json.loads(sample_typedef_json)))

    f2({"Bar": [1, 2.0, "3", [1, 2.0, "3"]]})
