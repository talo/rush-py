# Generated by ariadne-codegen
# Source: gql

from .base_model import BaseModel
from .fragments import SmolConformerFields


class CreateSmolConformer(BaseModel):
    create_smol_conformer: "CreateSmolConformerCreateSmolConformer"


class CreateSmolConformerCreateSmolConformer(SmolConformerFields):
    pass


CreateSmolConformer.model_rebuild()
