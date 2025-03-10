# Generated by ariadne-codegen
# Source: gql

from uuid import UUID

from .base_model import BaseModel
from .enums import ModuleInstanceStatus


class UpdateModuleInstance(BaseModel):
    update_module_instance: "UpdateModuleInstanceUpdateModuleInstance"


class UpdateModuleInstanceUpdateModuleInstance(BaseModel):
    id: UUID
    status: ModuleInstanceStatus


UpdateModuleInstance.model_rebuild()
