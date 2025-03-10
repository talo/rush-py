# Generated by ariadne-codegen
# Source: gql

from .base_model import BaseModel
from .fragments import BindingPoseConformerInteractionFields


class BindingPoseConformerInteraction(BaseModel):
    me: "BindingPoseConformerInteractionMe"


class BindingPoseConformerInteractionMe(BaseModel):
    account: "BindingPoseConformerInteractionMeAccount"


class BindingPoseConformerInteractionMeAccount(BaseModel):
    project: "BindingPoseConformerInteractionMeAccountProject"


class BindingPoseConformerInteractionMeAccountProject(BaseModel):
    binding_pose_conformer_interaction: "BindingPoseConformerInteractionMeAccountProjectBindingPoseConformerInteraction"


class BindingPoseConformerInteractionMeAccountProjectBindingPoseConformerInteraction(
    BindingPoseConformerInteractionFields
):
    pass


BindingPoseConformerInteraction.model_rebuild()
BindingPoseConformerInteractionMe.model_rebuild()
BindingPoseConformerInteractionMeAccount.model_rebuild()
BindingPoseConformerInteractionMeAccountProject.model_rebuild()
