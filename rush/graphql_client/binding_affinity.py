# Generated by ariadne-codegen
# Source: gql

from .base_model import BaseModel
from .fragments import BindingAffinityFields


class BindingAffinity(BaseModel):
    me: "BindingAffinityMe"


class BindingAffinityMe(BaseModel):
    account: "BindingAffinityMeAccount"


class BindingAffinityMeAccount(BaseModel):
    project: "BindingAffinityMeAccountProject"


class BindingAffinityMeAccountProject(BaseModel):
    binding_affinity: "BindingAffinityMeAccountProjectBindingAffinity"


class BindingAffinityMeAccountProjectBindingAffinity(BindingAffinityFields):
    pass


BindingAffinity.model_rebuild()
BindingAffinityMe.model_rebuild()
BindingAffinityMeAccount.model_rebuild()
BindingAffinityMeAccountProject.model_rebuild()
