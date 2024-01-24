"""Main entrypoint into package."""
from .provider import Provider, build_provider_with_functions
from .local import LocalProvider
from .protocols import run_qp
from .graphql_client.input_types import ModuleInstanceResourcesInput as Resources

Arg = Provider.Arg

__all__ = ["LocalProvider", "Provider", "run_qp", "Resources", "build_provider_with_functions", "Arg"]
