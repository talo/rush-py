# Generated by ariadne-codegen
# Source: gql

from typing import Any

from .base_model import BaseModel


class DeleteBindingSiteConformer(BaseModel):
    delete_binding_site_conformer: "DeleteBindingSiteConformerDeleteBindingSiteConformer"


class DeleteBindingSiteConformerDeleteBindingSiteConformer(BaseModel):
    id: Any


DeleteBindingSiteConformer.model_rebuild()
