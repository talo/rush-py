from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from io import BytesIO, StringIO
from numbers import Number
from pathlib import Path
from typing import Any, Generic, Literal, Tuple, TypeVar, Union
from uuid import UUID

from rush.graphql_client.upload_large_object import UploadLargeObjectUploadLargeObjectDescriptor

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


T = TypeVar("T")


if sys.version_info >= (3, 12):
    from .types import Conformer, EnumValue, Optional, Record, RushObject

else:
    from typing import Optional

    from .legacy_types import Conformer, EnumValue, Record, RushObject


U = TypeVar("U", bytes, Conformer, Record, list[Any], float)


class _RushObject(Generic[U]):
    object: U | None = None


SCALARS = Literal[
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
    "a3m",
    "gro",
    "mdp",
    "mol2",
    "pdb",
    "pdbqt",
    "sdf",
    "smi",
    "trr",
    "xtc",
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
    "a3m",
    "gro",
    "mdp",
    "mol2",
    "pdb",
    "pdbqt",
    "sdf",
    "smi",
    "trr",
    "xtc",
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
    "a3m": bytes,
    "gro": bytes,
    "mdp": bytes,
    "mol2": bytes,
    "pdb": bytes,
    "pdbqt": bytes,
    "sdf": bytes,
    "smi": bytes,
    "trr": bytes,
    "xtc": bytes,
}

KINDS = Literal["array", "fallible", "λ", "optional", "enum", "record", "tuple", "@", "union"]


@dataclass
class SimpleType:
    k: KINDS | None
    t: SCALARS | "SimpleType"


class RushType(Generic[T]):
    def __init__(
        self,
        type: "dict[str, RushType[T]] | list[RushType[T]] | RushType[T]",
        kind: KINDS | None = None,
        name: str | None = None,
        doc: str | None = None,
    ):
        self.k = kind
        self.t = type
        self.n = name
        self.doc = doc

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
            if any([x.lower().replace("_", "") == other.lower().replace("_", "") for x in self.literals]):
                return (True, None)
            if any(
                [
                    list(x.keys())[0].lower().replace("_", "") == other.lower().replace("_", "")
                    for x in self.tags
                ]
            ):
                return (True, None)
        return (False, f"Unknown enum variant, not list or dict {other}")


class RecordKind(Generic[T], RushType[T]):
    def __init__(self, record: dict[str, RushType[T]] | tuple[RushType[T]]):
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


class UnionKind(Generic[T], RushType[T]):
    def __init__(self, record: dict[str, RushType[T]] | tuple[RushType[T]]):
        self.k = "union"
        self.t = record

    def to_python_type(self) -> type[Union[Any, Any]]:
        return Union[Any, Any]

    def matches(self, other: dict[str, Any] | Any) -> tuple[bool, str | None]:
        if not isinstance(other, dict):
            return (False, f"Expected dict, got {type(other)}")
        # handle tuple case
        if (
            len(other) == 1
            and isinstance(list(other.values())[0], (list, tuple))
            and isinstance(self.t, tuple)
        ):
            other = list(other.values())[0]
            return self.t[0].matches(other)

        if isinstance(self.t, dict):
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
        if not isinstance(other, (list, tuple)):
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


class FallibleKind(Generic[T], RushType[T]):
    def __init__(self, fallible: RushType[T]):
        self.k = "fallible"
        self.t = fallible

    def to_python_type(self) -> type[Any]:
        return Tuple[Optional[self.t.to_python_type()], Optional[str]]

    def matches(self, other: Tuple[Optional[T], Optional[str]] | Any) -> tuple[bool, str | None]:
        if not isinstance(other, (list, tuple)):
            return (False, f"Expected list or tuple, got {type(other)}")
        # TODO: actually implement checking fallibles.
        #       in most cases, users will unwrap before passing as arguments so this is low prio
        return (True, None)


class ObjectKind(Generic[T], RushType[T]):
    def __init__(self, object: RushType[T]):
        self.k = "record"
        self.t = object
        self.n = "Object"

    def to_python_type(self) -> type[RushObject[Any]]:
        return RushObject[self.t.to_python_type()]

    def matches(self, other: Path | StringIO | BytesIO | Any) -> tuple[bool, str | None]:
        if isinstance(
            other,
            (
                _RushObject,
                Path,
                StringIO,
                UploadLargeObjectUploadLargeObjectDescriptor,
            ),
        ):
            return (True, None)
        else:
            return (False, f"Expected Path, got {type(other)}")


class LiteralKind(Generic[T], RushType[T]):
    def __init__(self, literal: T):
        self.k = None
        self.t = literal

    def to_python_type(self) -> type[Any]:
        return type(self.t)

    def matches(self, other: T) -> tuple[bool, str | None]:
        if other == self.t:
            return (True, None)
        else:
            return (False, f"Expected {self.t}, got {other}")


class ScalarType(Generic[T], RushType[T]):
    def __init__(self, scalar: SCALARS | str | int):
        self.k = None
        self.t = scalar
        if self.t not in SCALAR_STRS and not isinstance(self.t, int):
            self.t = self.t.replace("$", "").lower()
        self.py_type = scalar_types_mapping.get(self.t) if isinstance(self.t, str) else int
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
        elif isinstance(other, self.py_type) or (isinstance(other, Number) and self.py_type is float):
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
            elif res["k"] == "record" and res["n"] == "Object":
                return ObjectKind(type_from_typedef(res["t"]["path"]["t"]))
            elif res["k"] == "record":
                if isinstance(res["t"], dict):
                    return RecordKind({k: type_from_typedef(v) for k, v in res["t"].items()})
                else:
                    return RecordKind(tuple(type_from_typedef(v) for v in res["t"]))
            elif res["k"] == "union":
                if isinstance(res["t"], dict):
                    return UnionKind({k: type_from_typedef(v) for k, v in res["t"].items()})
                else:
                    return UnionKind(tuple(type_from_typedef(list(v.items())[0]) for v in res["t"]))
            elif res["k"] == "array":
                return ArrayKind(type_from_typedef(res["t"]))
            elif res["k"] == "tuple":
                return TupleKind(tuple(type_from_typedef(x) for x in res["t"]))
            elif res["k"] == "optional":
                return OptionalKind(type_from_typedef(res["t"]))
            elif res["k"] == "@":
                return type_from_typedef(res["t"])
            elif res["k"] == "alias":
                return type_from_typedef(res["t"])
            elif res["k"] == "fallible":
                return FallibleKind(type_from_typedef(res["t"]))
            else:
                raise Exception(f"Unknown kind {res['k']}")
        else:
            if res.get("t"):
                return type_from_typedef(res["t"])
            else:  # assuming typed enum
                return EnumKind([{k: type_from_typedef(v)} for k, v in res.items()])
    elif isinstance(res, (list, tuple)):
        return TupleKind([type_from_typedef(x) for x in res])
    elif isinstance(res, str):  # type: ignore
        return ScalarType(res)
    elif isinstance(res, int):  # type: ignore
        return LiteralKind(res)
    else:
        print("Bad type!", res)
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
                    continue
                else:
                    raise Exception(f"Typecheck failed: {match[1]}")

    return built


def format_module_typedesc(typedesc_in: str) -> str:
    def format_typedesc_line(old_line: str) -> list[str]:
        new_lines = []
        seen_nester = False
        nester_char = None
        seen_dict = {"{}": 0, "()": 0}
        last_break_pos = None
        good_nesting_level = False
        leading_spaces = " " * (len(old_line) - len(old_line.lstrip(" ")))
        for i, char in enumerate(old_line):
            if char in "{(":
                if not seen_nester:
                    seen_nester = True
                    nester_char = char
                    new_lines += [old_line[: i + 1]]
                    last_break_pos = i + 1
                seen_dict["{}" if char in "{}" else "()"] += 1
            if char in "})":
                seen_dict["{}" if char in "{}" else "()"] -= 1
            if seen_nester:
                if nester_char == "{":
                    good_nesting_level = seen_dict["{}"] == 1 and seen_dict["()"] == 0
                elif nester_char == "(":
                    good_nesting_level = seen_dict["()"] == 1 and seen_dict["{}"] == 0
                else:
                    print("ERROR!")
            if seen_nester and good_nesting_level and char == ",":
                new_lines += [leading_spaces + "    " + old_line[last_break_pos : i + 1].lstrip(" ")]
                last_break_pos = i + 1
            if seen_nester and seen_dict["{}"] == 0 and seen_dict["()"] == 0:
                # breaks in union
                # assert char in "})"
                new_lines += [leading_spaces + "    " + old_line[last_break_pos:i].lstrip(" ")]
                new_lines += [leading_spaces + old_line[i:].lstrip(" ")]
                break
        return new_lines

    old_lines = typedesc_in.replace(";", ";\n").replace("-> ", "\n->\n").split("\n")
    old_lines = ["    " + line.strip() for line in old_lines]
    some_line_too_long = True
    new_lines: list[str] = []
    while some_line_too_long:
        some_line_too_long = False
        new_lines = []
        for line in old_lines:
            if len(line) > 88:
                some_line_too_long = True
                new_lines += format_typedesc_line(line)
            else:
                new_lines += [line]
        old_lines = new_lines

    new_lines = [
        line.replace("{", " {").replace(",", ", ").replace(":", ": ").replace("|", " | ")
        for line in new_lines
    ]

    finalized_str = "\n".join([line.rstrip() for line in new_lines])
    finalized_str = re.sub(r", +", ", ", finalized_str)
    finalized_str = re.sub(r": +", ": ", finalized_str)
    finalized_str = re.sub(r" +\|", " |", finalized_str)
    finalized_str = re.sub(r"\| +", "| ", finalized_str)
    # TODO: render object properly and remove this hack
    finalized_str = re.sub(r" \{path: (.*?), size: (.*?)\ .*}", r"[\1]", finalized_str)
    finalized_str = re.sub(r" \{size: (.*?), path: (.*?)\ .*}", r"[\2]", finalized_str)

    return finalized_str + "\n"
