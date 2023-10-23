#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Any, Literal, Union
import typing
import uuid

SCALARS = Literal[
    "bool", "u8", "u16", "u32", "u64", "i8", "i16", "i32", "i64", "f32", "f64", "string", "bytes", "Conformer"
]


def py_scalar_type(scalar: SCALARS):
    if scalar == "bool":
        return bool
    elif (
        scalar == "u8"
        or scalar == "i8"
        or scalar == "u16"
        or scalar == "i16"
        or scalar == "u32"
        or scalar == "i32"
        or scalar == "u64"
        or scalar == "i64"
    ):
        return int
    elif scalar == "f32" or scalar == "f64":
        return float
    elif scalar == "string":
        return str
    elif scalar == "bytes":
        return bytes
    elif scalar == "Conformer":
        return dict


KINDS = Literal["array", "optional", "enum", "record", "object", "tuple"]


class Type:
    def __init__(self, type: Union[dict[str, "Type"], list["Type"], "Type"], kind: KINDS | None = None):
        self.k: KINDS | None = kind
        self.t = type

    def to_json(self):
        if self.k is None:
            return self.t
        else:
            inner_ts = self.t.to_json() if isinstance(self.t, Type) else self.t
            if isinstance(inner_ts, list):
                inner_ts = [x.to_json() if isinstance(x, Type) else x for x in inner_ts]
            elif isinstance(inner_ts, dict):
                inner_ts = {k: v.to_json() if isinstance(v, Type) else v for k, v in inner_ts.items()}
            return {"k": self.k, "t": inner_ts}

    def to_python_type(self) -> typing.Type[Any]:
        match self.k:
            case "array" | "tuple":
                if isinstance(self.t, list):
                    return list[self.t[0].to_python_type()]
                if isinstance(self.t, Type):
                    return list[self.t.to_python_type()]
                else:
                    return list[Any]
            case "optional":
                if isinstance(self.t, list):
                    return list[self.t[0].to_python_type()] | None
                if isinstance(self.t, Type):
                    return list[self.t.to_python_type()] | None
                else:
                    return list[Any] | None
            case "enum":
                return dict[str, Any] | str
            case "record":
                return dict[str, Any]
            case "object":
                return Path
            case None:
                if isinstance(self.t, list):
                    return list
                else:
                    return dict

    def matches(self, other: Any) -> tuple[bool, str | None]:
        """
        Compare a Type to a python object. Returns true if the object matches the type.
        """
        # TODO: we might want to fetch remote types here
        if isinstance(other, uuid.UUID):
            return (True, None)

        if self.k is None:
            if isinstance(self.t, list) or isinstance(self.t, tuple):
                if not (isinstance(other, list) or isinstance(other, tuple)):
                    print(f"Expected list or tuple, got {type(other)}")
                    reason = f"Expected list or tuple, got {type(other)}"
                    return (False, reason)
                for x, y in zip(self.t, other):
                    ok, reason = x.matches(y)
                    if not ok:
                        return (False, reason)
                return (True, None)
            elif isinstance(self.t, dict):
                if not isinstance(other, dict):
                    reason = f"Expected dict, got {type(other)}"
                    return (False, reason)
                for k, v in self.t.items():
                    if k not in other:
                        reason = f"Expected key {k} in dict"
                        return (False, reason)
                    ok, reason = v.matches(other[k])
                    if not ok:
                        return (False, reason)
                return (True, None)
            elif isinstance(self.t, ScalarType):
                return self.t.matches(other)
            return self.t.matches(other)
            # else:
            #     raise Exception(f"Invalid type: {self.t.to_json()} can not be checked against {other}")
        elif self.k == "array":
            if not (isinstance(other, list) or isinstance(other, tuple)):
                return (False, f"Expected list or tuple, got {type(other)}")
            for x in other:
                ok, reason = Type(self.t).matches(x)
                if not ok:
                    return (False, reason)
            return (True, None)
        elif self.k == "object":
            if isinstance(other, Path):
                return (True, None)
            else:
                return (False, f"Expected Path, got {type(other)}")
        elif self.k == "tuple":
            if not (isinstance(other, list) or isinstance(other, tuple)):
                return (False, f"Expected list or tuple, got {type(other)}")
            if len(other) != len(self.t.t):
                return (False, f"Expected list of length {len(self.t.t)}, got {len(other)}")
            for x, y in zip(self.t.t, other):
                ok, reason = x.matches(y)
                if not ok:
                    return (False, reason)
            return (True, None)
        elif self.k == "optional":
            if other is None:
                return (True, None)
            else:
                return (self.t.t if isinstance(self.t, dict) else self.t).matches(other)
                # return self.t.t.matches(other)
        elif self.k == "enum":
            if isinstance(other, dict) and len(other) == 1:
                ok = list(other.keys())[0]
                for t in self.t.t:
                    if isinstance(t.t, str) and t.t == ok:
                        return (True, None)
                    elif isinstance(t.t, dict) and list(t.t.keys())[0] == ok:
                        return t.t[ok].matches(other[ok])
                return (False, f"Unknown enum variant {ok}")
            if isinstance(other, str):
                if any([x.matches(other) for x in self.t.t]):
                    return (True, None)
                else:
                    return (False, f"Unknown enum variant {other}")

        elif self.k == "record":
            if isinstance(other, dict):
                for k, v in (self.t if isinstance(self.t, dict) else self.t.t).items():
                    if k not in other:
                        return (False, f"Expected key {k} in dict")
                    ok, reason = v.matches(other[k])
                    if not ok:
                        return (False, reason)
                return (True, None)
            else:
                return (False, f"Expected dict, got {type(other)}")
        else:
            raise Exception("Invalid type")

        raise Exception("Invalid type")


class ScalarType(Type):
    def __init__(self, scalar: SCALARS | "str"):
        self.k = None
        self.t = self  # scalar
        self.r = py_scalar_type(scalar)
        self.l = scalar if not self.r else None

    def to_python_type(self):
        return self.r

    def matches(self, other: Any) -> tuple[bool, str | None]:
        if self.l:
            if other == self.l:
                return (True, None)
            else:
                return (False, f"Expected {self.l}, got {other}")
        if isinstance(other, self.r):
            return (True, None)
        else:
            return (False, f"Expected {self.r}, got {other}")


def type_from_typedef(res: Any) -> Type:
    if isinstance(res, dict):
        if res.get("t") and res.get("k"):
            return (
                # ScalarType(res["t"])
                # if isinstance(res["t"], str)
                Type(type_from_typedef(res["t"]), res["k"])
            )
        else:
            return Type({k: type_from_typedef(v) for k, v in res.items()})
    elif isinstance(res, list):
        return Type([type_from_typedef(x) for x in res])
    else:
        return ScalarType(res) if isinstance(res, str) else Type(res)


sample_typedef_json = '{"k":"enum","t":["Foo",{"Bar":["i32","f32","string",{"k":"tuple","t":["i32","f32","string"]}]},{"Baz":{"f":"f32","i":"i32","s":"string","t":{"k":"tuple","t":["i32","f32","string"]}}}]}'


def build_function_with_typedef(types: list[Type], docs: str):
    def built(*args: Any):
        for t, a in zip(types, args):
            match = t.matches(a)
            if not match[0]:
                raise Exception(f"Typecheck failed: {match[1]}")

    built.__doc__ = docs

    return built


def test():
    f = build_function_with_typedef(
        [
            Type(ScalarType("bytes"), "object"),
            Type({"a": ScalarType("i32")}, "record"),
            Type({"b": ScalarType("i32")}, "record"),
        ],
        "docs",
    )
    f(Path("."), {"a": 1}, {"b": 2})

    f2 = build_function_with_typedef([type_from_typedef(json.loads(sample_typedef_json))], "docs")

    f2({"Bar": [1, 2.0, "3", [1, 2.0, "3"]]})
