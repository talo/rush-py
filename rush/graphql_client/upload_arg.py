# Generated by ariadne-codegen
# Source: combined.graphql

from typing import Any, Optional
from uuid import UUID

from .base_model import BaseModel


class UploadArg(BaseModel):
    upload_arg: "UploadArgUploadArg"


class UploadArgUploadArg(BaseModel):
    id: UUID
    value: Optional[Any]


UploadArg.model_rebuild()
