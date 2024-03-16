#!/usr/bin/env python3
import sys
from typing import Any
from uuid import UUID

from .graphql_client.enums import ModuleInstanceTarget
from .graphql_client.input_types import ModuleInstanceResourcesInput

if sys.version_info >= (3, 12):
    type Target = ModuleInstanceTarget | str
    type Resources = ModuleInstanceResourcesInput | dict[str, Any]
    ArgId = UUID
    ModuleInstanceId = UUID
else:
    try:
        from typing import TypeAlias
    except ImportError:
        from typing_extensions import TypeAlias
    try:
        from typing import TypeAliasType
    except ImportError:
        from typing_extensions import TypeAliasType
    ArgId: TypeAlias = UUID
    ModuleInstanceId: TypeAlias = UUID
    if sys.version_info >= (3, 10):
        Target = TypeAliasType("Target", ModuleInstanceTarget | str)
        Resources = TypeAliasType("Resources", ModuleInstanceResourcesInput | dict[str, Any])
    else:
        Target = TypeAliasType("Target", Union[ModuleInstanceTarget, str])
        Resources = TypeAliasType("Resources", Union[ModuleInstanceResourcesInput, dict[str, Any]])
