# Generated by ariadne-codegen
# Source: combined.graphql

from datetime import datetime
from typing import Any, List, Optional

from pydantic import Field

from .base_model import BaseModel


class Entity(BaseModel):
    entity: Optional["EntityEntity"]


class EntityEntity(BaseModel):
    id: Any
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    deleted_at: Optional[datetime] = Field(alias="deletedAt")
    data: Optional[Any]
    tags: List[str]
