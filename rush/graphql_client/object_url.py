# Generated by ariadne-codegen
# Source: gql

from typing import Optional

from .base_model import BaseModel


class ObjectUrl(BaseModel):
    object_path: "ObjectUrlObjectPath"


class ObjectUrlObjectPath(BaseModel):
    url: Optional[str]


ObjectUrl.model_rebuild()
