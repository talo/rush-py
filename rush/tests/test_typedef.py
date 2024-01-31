import json
from pathlib import Path
from rush import typedef

from rush.typedef import ObjectKind, ScalarType, RecordKind, Literal, build_typechecker

sample_typedef_json = """{
  "k": "enum",
  "t": [
    "Foo",
    {
      "Bar": [
        "i32",
        "f32",
        "string",
        {
          "k": "tuple",
          "t": [
            "i32",
            "f32",
            "string"
          ]
        }
      ]
    },
    {
      "Baz": {
        "f": "f32",
        "i": "i32",
        "s": "string",
        "t": {
          "k": "tuple",
          "t": [
            "i32",
            "f32",
            "string"
          ]
        }
      }
    }
  ]
}"""


def test_typecheck():
    f = build_typechecker(
        ObjectKind(ScalarType[Literal["bytes"]]("bytes")),
        RecordKind({"a": ScalarType[Literal["i32"]]("i32")}),
        RecordKind({"b": ScalarType("i32")}),
    )
    f(Path("."), {"a": 1}, {"b": 2})

    f2 = build_typechecker(typedef.type_from_typedef(json.loads(sample_typedef_json)))

    f2({"Bar": [1, 2.0, "3", [1, 2.0, "3"]]})
