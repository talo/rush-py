# Generated by ariadne-codegen
# Source: gql

from .base_model import BaseModel
from .fragments import SmolLibraryPartitionFields


class PatchSmolLibraryPartition(BaseModel):
    patch_smol_library_partition: "PatchSmolLibraryPartitionPatchSmolLibraryPartition"


class PatchSmolLibraryPartitionPatchSmolLibraryPartition(SmolLibraryPartitionFields):
    pass


PatchSmolLibraryPartition.model_rebuild()
