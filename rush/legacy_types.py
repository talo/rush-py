from __future__ import annotations

import sys
from typing import Any, Generic, TypeVar, Union
from uuid import UUID

from .graphql_client.enums import ModuleInstanceTarget
from .graphql_client.input_types import ModuleInstanceResourcesInput

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

try:
    from typing import TypeAliasType
except ImportError:
    from typing_extensions import TypeAliasType

if sys.version_info >= (3, 10):
    Target = TypeAliasType("Target", ModuleInstanceTarget | str)
    Resources = TypeAliasType("Resources", ModuleInstanceResourcesInput | dict[str, Any])
else:
    Target = TypeAliasType("Target", Union[ModuleInstanceTarget, str])
    Resources = TypeAliasType("Resources", Union[ModuleInstanceResourcesInput, dict[str, Any]])

ArgId: TypeAlias = UUID
ModuleInstanceId: TypeAlias = UUID

Conformer = TypeAliasType("Conformer", dict[str, Any])
Record = TypeAliasType("Record", dict[str, Any])
EnumValue = TypeAliasType("EnumValue", str)

U = TypeVar("U", bytes, Conformer, Record, list[Any], float)


class _RushObject(Generic[U]):
    object: U | None = None


RushObject = TypeAliasType("RushObject", _RushObject[U], type_params=(U,))
