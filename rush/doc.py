#!/usr/bin/env python3
import asyncio
from typing import Any, TypeAlias, TypeVar

from typing_extensions import TypeAliasType

from . import provider, typedef

Conformer: TypeAlias = dict[str, Any]
""" A Conformer represents a biochemical structure. It is documented in:  
https://talo.github.io/qdx-common/qdx_common/conformer/struct.Conformer.html
"""  # noqa: W291

Record: TypeAlias = dict[str, Any]
""" A `dictionary` representing JSON data.

As an input, it usually stores configuration info.

As an output, it usually stores data containing results from the corresponding run.
"""

EnumValue: TypeAlias = str
""" A QDX Enum type. From python, pass a `str`."""

T = TypeVar("T", bytes, Conformer, Record, list[Any], float)
_RushObject = TypeAliasType("_RushObject", typedef.RushObject[T], type_params=(T,))
RushObject: TypeAlias = _RushObject[T]
""" Represents a file in object storage.

As an input argument, pass
an object uploaded or obtained from another run, a `pathlib.Path`, or a `StringIO`.

As an output argument, you can expect:
  - a URL for non-JSON files,
  - a `Conformer` or `Record` (i.e. `dict`) for JSON files, or
  - the expected containers or values for other types (`list`, `float`, et cetera).
"""

_x = provider.Provider()
_fns = asyncio.run(_x.get_module_functions(tags=["rush-py-v1.4.0"]))
RushProvider = type(
    "RushProvider",
    (provider.Provider,),
    _fns | {n: getattr(_x, n) for n in dir(_x)},
)
