"""Main entrypoint into package."""

from .graphql_client.input_types import ResourcesInput as Resources
#from .local import LocalProvider
from .provider import Provider, build_blocking_provider, build_blocking_provider_with_functions, build_provider_with_functions

Arg = Provider.Arg

__all__ = [
    #"LocalProvider",
    "Provider",
    "Resources",
    "build_blocking_provider_with_functions",
    "build_provider_with_functions",
    "build_blocking_provider",
    "Arg",
]
