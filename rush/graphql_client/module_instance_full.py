# Generated by ariadne-codegen
# Source: combined.graphql

from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import Field

from .base_model import BaseModel
from .enums import ModuleInstanceStatus, ModuleInstanceTarget


class ModuleInstanceFull(BaseModel):
    module_instance: "ModuleInstanceFullModuleInstance"


class ModuleInstanceFullModuleInstance(BaseModel):
    id: UUID
    created_at: datetime
    deleted_at: Optional[datetime]
    account_id: UUID
    path: str
    ins: List["ModuleInstanceFullModuleInstanceIns"]
    outs: List["ModuleInstanceFullModuleInstanceOuts"]
    queued_at: Optional[datetime]
    admitted_at: Optional[datetime]
    dispatched_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: ModuleInstanceStatus
    target: ModuleInstanceTarget
    resources: Optional["ModuleInstanceFullModuleInstanceResources"]
    progress: Optional["ModuleInstanceFullModuleInstanceProgress"]
    tags: Optional[List[str]]
    stdout: "ModuleInstanceFullModuleInstanceStdout"
    stderr: "ModuleInstanceFullModuleInstanceStderr"


class ModuleInstanceFullModuleInstanceIns(BaseModel):
    id: UUID
    created_at: datetime
    deleted_at: Optional[datetime]
    rejected_at: Optional[datetime]
    account_id: UUID
    typeinfo: Any
    value: Optional[Any]
    tags: Optional[List[str]]


class ModuleInstanceFullModuleInstanceOuts(BaseModel):
    id: UUID
    created_at: datetime
    deleted_at: Optional[datetime]
    rejected_at: Optional[datetime]
    account_id: UUID
    typeinfo: Any
    value: Optional[Any]
    tags: Optional[List[str]]


class ModuleInstanceFullModuleInstanceResources(BaseModel):
    gpus: Optional[int]
    nodes: Optional[int]
    mem: Optional[int]
    storage: Optional[int]
    walltime: Optional[int]


class ModuleInstanceFullModuleInstanceProgress(BaseModel):
    n: int
    n_expected: int
    n_max: int
    done: bool


class ModuleInstanceFullModuleInstanceStdout(BaseModel):
    page_info: "ModuleInstanceFullModuleInstanceStdoutPageInfo" = Field(alias="pageInfo")
    edges: List["ModuleInstanceFullModuleInstanceStdoutEdges"]


class ModuleInstanceFullModuleInstanceStdoutPageInfo(BaseModel):
    has_previous_page: bool = Field(alias="hasPreviousPage")
    has_next_page: bool = Field(alias="hasNextPage")
    start_cursor: Optional[str] = Field(alias="startCursor")
    end_cursor: Optional[str] = Field(alias="endCursor")


class ModuleInstanceFullModuleInstanceStdoutEdges(BaseModel):
    cursor: str
    node: "ModuleInstanceFullModuleInstanceStdoutEdgesNode"


class ModuleInstanceFullModuleInstanceStdoutEdgesNode(BaseModel):
    id: str
    created_at: datetime
    content: List[str]


class ModuleInstanceFullModuleInstanceStderr(BaseModel):
    page_info: "ModuleInstanceFullModuleInstanceStderrPageInfo" = Field(alias="pageInfo")
    edges: List["ModuleInstanceFullModuleInstanceStderrEdges"]


class ModuleInstanceFullModuleInstanceStderrPageInfo(BaseModel):
    has_previous_page: bool = Field(alias="hasPreviousPage")
    has_next_page: bool = Field(alias="hasNextPage")
    start_cursor: Optional[str] = Field(alias="startCursor")
    end_cursor: Optional[str] = Field(alias="endCursor")


class ModuleInstanceFullModuleInstanceStderrEdges(BaseModel):
    cursor: str
    node: "ModuleInstanceFullModuleInstanceStderrEdgesNode"


class ModuleInstanceFullModuleInstanceStderrEdgesNode(BaseModel):
    id: str
    created_at: datetime
    content: List[str]


ModuleInstanceFull.model_rebuild()
ModuleInstanceFullModuleInstance.model_rebuild()
ModuleInstanceFullModuleInstanceIns.model_rebuild()
ModuleInstanceFullModuleInstanceOuts.model_rebuild()
ModuleInstanceFullModuleInstanceResources.model_rebuild()
ModuleInstanceFullModuleInstanceProgress.model_rebuild()
ModuleInstanceFullModuleInstanceStdout.model_rebuild()
ModuleInstanceFullModuleInstanceStdoutPageInfo.model_rebuild()
ModuleInstanceFullModuleInstanceStdoutEdges.model_rebuild()
ModuleInstanceFullModuleInstanceStdoutEdgesNode.model_rebuild()
ModuleInstanceFullModuleInstanceStderr.model_rebuild()
ModuleInstanceFullModuleInstanceStderrPageInfo.model_rebuild()
ModuleInstanceFullModuleInstanceStderrEdges.model_rebuild()
ModuleInstanceFullModuleInstanceStderrEdgesNode.model_rebuild()
