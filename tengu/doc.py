#!/usr/bin/env python3
from gql.transport.requests import RequestsHTTPTransport

from . import api

_x = api.BaseTypedProvider(RequestsHTTPTransport(url="https://tengu.qdx.ai"))

TenguProvider = type(
    "TenguProvider",
    (api.BaseTypedProvider,),
    _x.get_module_functions() | {n: getattr(_x, n) for n in dir(_x)},
)
