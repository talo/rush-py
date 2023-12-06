#!/usr/bin/env python3
import asyncio

from . import provider

_x = provider.Provider()
fns = asyncio.run(_x.get_module_functions())
TenguProvider = type(
    "TenguProvider",
    (provider.Provider,),
    fns | {n: getattr(_x, n) for n in dir(_x)},
)
