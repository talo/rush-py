# Generated by ariadne-codegen
# Source: gql

from .base_model import BaseModel
from .fragments import BindingSiteConformerFields


class PatchBindingSiteConformer(BaseModel):
    patch_binding_site_conformer: "PatchBindingSiteConformerPatchBindingSiteConformer"


class PatchBindingSiteConformerPatchBindingSiteConformer(BindingSiteConformerFields):
    pass


PatchBindingSiteConformer.model_rebuild()
