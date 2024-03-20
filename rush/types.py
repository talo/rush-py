from typing import Any
from uuid import UUID

from .graphql_client.enums import ModuleInstanceTarget
from .graphql_client.input_types import ModuleInstanceResourcesInput

type Target = ModuleInstanceTarget | str
type Resources = ModuleInstanceResourcesInput | dict[str, Any]
ArgId = UUID
ModuleInstanceId = UUID
