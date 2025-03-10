# Generated by ariadne-codegen
# Source: gql

from typing import List

from pydantic import Field

from .base_model import BaseModel
from .fragments import BindingAffinityFields, PageInfoFull


class BindingAffinities(BaseModel):
    me: "BindingAffinitiesMe"


class BindingAffinitiesMe(BaseModel):
    account: "BindingAffinitiesMeAccount"


class BindingAffinitiesMeAccount(BaseModel):
    project: "BindingAffinitiesMeAccountProject"


class BindingAffinitiesMeAccountProject(BaseModel):
    binding_affinities: "BindingAffinitiesMeAccountProjectBindingAffinities"


class BindingAffinitiesMeAccountProjectBindingAffinities(BaseModel):
    page_info: "BindingAffinitiesMeAccountProjectBindingAffinitiesPageInfo" = Field(
        alias="pageInfo"
    )
    edges: List["BindingAffinitiesMeAccountProjectBindingAffinitiesEdges"]
    total_count: int


class BindingAffinitiesMeAccountProjectBindingAffinitiesPageInfo(PageInfoFull):
    pass


class BindingAffinitiesMeAccountProjectBindingAffinitiesEdges(BaseModel):
    cursor: str
    node: "BindingAffinitiesMeAccountProjectBindingAffinitiesEdgesNode"


class BindingAffinitiesMeAccountProjectBindingAffinitiesEdgesNode(
    BindingAffinityFields
):
    pass


BindingAffinities.model_rebuild()
BindingAffinitiesMe.model_rebuild()
BindingAffinitiesMeAccount.model_rebuild()
BindingAffinitiesMeAccountProject.model_rebuild()
BindingAffinitiesMeAccountProjectBindingAffinities.model_rebuild()
BindingAffinitiesMeAccountProjectBindingAffinitiesEdges.model_rebuild()
