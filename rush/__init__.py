"""Main entrypoint into package."""

from .graphql_client.input_types import ModuleInstanceResourcesInput as Resources
from .local import LocalProvider
from .protocols import run_qp
from .provider import Provider, build_blocking_provider_with_functions, build_provider_with_functions

Arg = Provider.Arg

__all__ = [
    "LocalProvider",
    "Provider",
    "run_qp",
    "Resources",
    "build_blocking_provider_with_functions",
    "build_provider_with_functions",
    "Arg",
]
