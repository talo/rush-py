# Generated by ariadne-codegen
# Source: gql

from .base_model import BaseModel
from .fragments import BindingPoseConformerFields


class BindingPoseConformer(BaseModel):
    me: "BindingPoseConformerMe"


class BindingPoseConformerMe(BaseModel):
    account: "BindingPoseConformerMeAccount"


class BindingPoseConformerMeAccount(BaseModel):
    project: "BindingPoseConformerMeAccountProject"


class BindingPoseConformerMeAccountProject(BaseModel):
    binding_pose_conformer: "BindingPoseConformerMeAccountProjectBindingPoseConformer"


class BindingPoseConformerMeAccountProjectBindingPoseConformer(
    BindingPoseConformerFields
):
    pass


BindingPoseConformer.model_rebuild()
BindingPoseConformerMe.model_rebuild()
BindingPoseConformerMeAccount.model_rebuild()
BindingPoseConformerMeAccountProject.model_rebuild()
