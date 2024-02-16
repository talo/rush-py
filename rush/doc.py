#!/usr/bin/env python3.12
# flake8: noqa
import asyncio
from typing import Any as _Any

from . import provider, typedef

type Any = _Any
"@private"

type Conformer = dict[str, Any]
""" A Conformer represents a biochemical structure. It is documented in:  
https://talo.github.io/qdx-common/qdx_common/conformer/struct.Conformer.html
"""  # noqa: W291

type Record = dict[str, Any]
""" A `dictionary` representing JSON data.

As an input, it usually stores configuration info.

As an output, it usually stores data containing results from the corresponding run.
"""

type EnumValue = str
""" A QDX Enum type. From python, pass a `str`."""

type RushObject = typedef.RushObject[Any]
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
