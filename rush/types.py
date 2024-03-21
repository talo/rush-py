from typing import Any, TypeVar
from typing import Optional as _Optional
from uuid import UUID

from .graphql_client.enums import ModuleInstanceTarget
from .graphql_client.input_types import ModuleInstanceResourcesInput

type Target = ModuleInstanceTarget | str
type Resources = ModuleInstanceResourcesInput | dict[str, Any]
ArgId = UUID
ModuleInstanceId = UUID

# For type styling in docs
type Optional[T] = _Optional[T]
type Conformer = dict[str, Any]
type Record = dict[str, Any]
type EnumValue = str

RushObjectTypes = bytes | Conformer | Record | list["RushObjectTypes"] | float
U = TypeVar("U", bytes, Conformer, Record, list[RushObjectTypes], float)


class _RushObject[U]:
    object: U | None = None


type RushObject[U] = _RushObject[U]
