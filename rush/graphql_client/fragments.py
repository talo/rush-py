# Generated by ariadne-codegen
# Source: gql

from datetime import datetime
from typing import Any, List, Literal, Optional
from uuid import UUID

from pydantic import Field

from .base_model import BaseModel
from .enums import (
    AccountTier,
    BenchmarkEntityType,
    BindingSiteInteractionKind,
    ModuleFailureReason,
    ModuleInstanceStatus,
    ModuleInstanceTarget,
    ObjectFormat,
    PiStackKind,
    RunStatus,
    TaggedType,
)


class AccountFields(BaseModel):
    id: UUID
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    tier: AccountTier
    config: Optional["AccountFieldsConfig"]


class AccountFieldsConfig(BaseModel):
    config_account: Optional[UUID]
    bucket_config: Optional["AccountFieldsConfigBucketConfig"]


class AccountFieldsConfigBucketConfig(BaseModel):
    data_bucket: str
    log_bucket: str
    bucket_region: str


class ArgumentPartial(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    name: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]
    rejected_at: Optional[datetime]
    account_id: UUID
    typeinfo: Any
    value: Optional[Any]
    source: Optional[UUID]


class BenchmarkDataFields(BaseModel):
    typename__: str = Field(alias="__typename")
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    inputs: List["BenchmarkDataFieldsInputs"]
    outputs: List["BenchmarkDataFieldsOutputs"]
    input_entities: List["BenchmarkDataFieldsInputEntities"]
    output_entities: List["BenchmarkDataFieldsOutputEntities"]


class BenchmarkDataFieldsInputs(BaseModel):
    typename__: Literal["BenchmarkArg"] = Field(alias="__typename")
    id: Any
    entity: BenchmarkEntityType


class BenchmarkDataFieldsOutputs(BaseModel):
    typename__: Literal["BenchmarkArg"] = Field(alias="__typename")
    entity: BenchmarkEntityType
    id: Any


class BenchmarkDataFieldsInputEntities(BaseModel):
    typename__: Literal[
        "Account",
        "Benchmark",
        "BenchmarkData",
        "BenchmarkSubmission",
        "BenchmarkSubmissionData",
        "BindingAffinity",
        "BindingAffinityActivity",
        "BindingPoseAffinity",
        "BindingPoseConformer",
        "BindingPoseConformerInteractions",
        "BindingSiteConformer",
        "BindingSiteInteractions",
        "Chat",
        "Entity",
        "Message",
        "Module",
        "ModuleInstance",
        "MultipleSequenceAlignment",
        "Paper",
        "PaperContent",
        "Project",
        "Protein",
        "ProteinConformer",
        "Run",
        "SarModel",
        "SarProgram",
        "Smol",
        "SmolConformer",
        "SmolLibrary",
        "SmolLibraryPartition",
        "Structure",
        "Tag",
        "Token",
        "User",
    ] = Field(alias="__typename")


class BenchmarkDataFieldsOutputEntities(BaseModel):
    typename__: Literal[
        "Account",
        "Benchmark",
        "BenchmarkData",
        "BenchmarkSubmission",
        "BenchmarkSubmissionData",
        "BindingAffinity",
        "BindingAffinityActivity",
        "BindingPoseAffinity",
        "BindingPoseConformer",
        "BindingPoseConformerInteractions",
        "BindingSiteConformer",
        "BindingSiteInteractions",
        "Chat",
        "Entity",
        "Message",
        "Module",
        "ModuleInstance",
        "MultipleSequenceAlignment",
        "Paper",
        "PaperContent",
        "Project",
        "Protein",
        "ProteinConformer",
        "Run",
        "SarModel",
        "SarProgram",
        "Smol",
        "SmolConformer",
        "SmolLibrary",
        "SmolLibraryPartition",
        "Structure",
        "Tag",
        "Token",
        "User",
    ] = Field(alias="__typename")


class BenchmarkFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]


class BenchmarkSubmissionFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    scores: "BenchmarkSubmissionFieldsScores"
    benchmark: "BenchmarkSubmissionFieldsBenchmark"
    data: "BenchmarkSubmissionFieldsData"
    source_run: Optional["BenchmarkSubmissionFieldsSourceRun"]


class BenchmarkSubmissionFieldsScores(BaseModel):
    nodes: List["BenchmarkSubmissionFieldsScoresNodes"]


class BenchmarkSubmissionFieldsScoresNodes(BaseModel):
    id: Any
    score: float
    name: Optional[str]
    tags: Optional[List[str]]


class BenchmarkSubmissionFieldsBenchmark(BaseModel):
    id: Any


class BenchmarkSubmissionFieldsData(BaseModel):
    nodes: List["BenchmarkSubmissionFieldsDataNodes"]


class BenchmarkSubmissionFieldsDataNodes(BaseModel):
    id: Any
    scores: "BenchmarkSubmissionFieldsDataNodesScores"


class BenchmarkSubmissionFieldsDataNodesScores(BaseModel):
    nodes: List["BenchmarkSubmissionFieldsDataNodesScoresNodes"]


class BenchmarkSubmissionFieldsDataNodesScoresNodes(BaseModel):
    id: Any
    score: float
    name: Optional[str]
    tags: Optional[List[str]]


class BenchmarkSubmissionFieldsSourceRun(BaseModel):
    id: Any
    status: RunStatus
    result: Optional[Any]


class BindingAffinityFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    affinity: float
    affinity_metric: str
    protein: "BindingAffinityFieldsProtein"
    smol: "BindingAffinityFieldsSmol"


class BindingAffinityFieldsProtein(BaseModel):
    id: Any
    sequence: str


class BindingAffinityFieldsSmol(BaseModel):
    id: Any
    smi: Optional[str]


class BindingPoseAffinityFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]


class BindingPoseConformerFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    smol_conformer: "BindingPoseConformerFieldsSmolConformer"
    binding_site_conformer: "BindingPoseConformerFieldsBindingSiteConformer"


class BindingPoseConformerFieldsSmolConformer(BaseModel):
    id: Any
    residues: List[int]


class BindingPoseConformerFieldsBindingSiteConformer(BaseModel):
    id: Any
    bounding_box: "BindingPoseConformerFieldsBindingSiteConformerBoundingBox"


class BindingPoseConformerFieldsBindingSiteConformerBoundingBox(BaseModel):
    min: Any
    max: Any


class BindingPoseConformerInteractionFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    binding_pose_conformer: "BindingPoseConformerInteractionFieldsBindingPoseConformer"
    binding_site_interactions: "BindingPoseConformerInteractionFieldsBindingSiteInteractions"


class BindingPoseConformerInteractionFieldsBindingPoseConformer(BaseModel):
    id: Any


class BindingPoseConformerInteractionFieldsBindingSiteInteractions(BaseModel):
    id: Any
    residues: List[int]
    interactions: List[
        "BindingPoseConformerInteractionFieldsBindingSiteInteractionsInteractions"
    ]


class BindingPoseConformerInteractionFieldsBindingSiteInteractionsInteractions(
    BaseModel
):
    kind: BindingSiteInteractionKind
    pi_stack_kind: Optional[PiStackKind]
    ligand_atom: Any
    receptor_atom: Any


class BindingSiteConformerFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    bounding_box: "BindingSiteConformerFieldsBoundingBox"
    surface_atoms: Optional[List[int]]
    protein_conformer: "BindingSiteConformerFieldsProteinConformer"


class BindingSiteConformerFieldsBoundingBox(BaseModel):
    min: Any
    max: Any


class BindingSiteConformerFieldsProteinConformer(BaseModel):
    id: Any
    residues: List[int]


class BindingSiteInteractionFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    residues: List[int]
    interactions: List["BindingSiteInteractionFieldsInteractions"]


class BindingSiteInteractionFieldsInteractions(BaseModel):
    kind: BindingSiteInteractionKind
    pi_stack_kind: Optional[PiStackKind]
    ligand_atom: Any
    receptor_atom: Any


class MSAFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    a_3_m: Any = Field(alias="a_3m")


class ModuleFull(BaseModel):
    id: UUID
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    path: str
    ins: List[Any]
    ins_usage: Optional[List[str]]
    outs: List[Any]
    outs_usage: Optional[List[str]]
    typedesc: str
    targets: List[ModuleInstanceTarget]
    resource_bounds: Optional["ModuleFullResourceBounds"]


class ModuleFullResourceBounds(BaseModel):
    gpu_min: Optional[int]
    gpu_max: Optional[int]
    gpu_hint: Optional[int]
    gpu_mem_min: Optional[int]
    gpu_mem_max: Optional[int]
    gpu_mem_hint: Optional[int]
    cpu_min: Optional[int]
    cpu_max: Optional[int]
    cpu_hint: Optional[int]
    node_min: Optional[int]
    node_max: Optional[int]
    node_hint: Optional[int]
    mem_min: Optional[int]
    mem_max: Optional[int]
    storage_min: Optional[int]
    storage_max: Optional[int]


class ModuleInstanceBase(BaseModel):
    id: UUID
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    account_id: UUID
    queued_at: Optional[datetime]
    admitted_at: Optional[datetime]
    dispatched_at: Optional[datetime]
    completed_at: Optional[datetime]
    path: str
    status: ModuleInstanceStatus
    target: ModuleInstanceTarget
    failure_reason: Optional[ModuleFailureReason]
    failure_context: Optional["ModuleInstanceBaseFailureContext"]


class ModuleInstanceBaseFailureContext(BaseModel):
    stdout: Optional[str]
    stderr: Optional[str]
    syserr: Optional[str]


class ModuleInstanceFull(ModuleInstanceBase):
    ins: List["ModuleInstanceFullIns"]
    outs: List["ModuleInstanceFullOuts"]
    resources: Optional["ModuleInstanceFullResources"]
    progress: Optional["ModuleInstanceFullProgress"]
    resource_utilization: Optional["ModuleInstanceFullResourceUtilization"]


class ModuleInstanceFullIns(ArgumentPartial):
    pass


class ModuleInstanceFullOuts(ArgumentPartial):
    pass


class ModuleInstanceFullResources(BaseModel):
    gpus: Optional[int]
    nodes: Optional[int]
    mem: Optional[int]
    storage: Optional[int]
    walltime: Optional[int]


class ModuleInstanceFullProgress(BaseModel):
    n: int
    n_expected: int
    n_max: int
    done: bool


class ModuleInstanceFullResourceUtilization(BaseModel):
    gpu: Optional[float]
    mem: Optional[float]
    storage: float
    walltime: float
    cputime: float
    inodes: float
    sus: Optional[int]


class ObjectFields(BaseModel):
    id: UUID
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    object: "ObjectFieldsObject"
    type_info: Any
    base_url: str
    url: Optional[str]


class ObjectFieldsObject(BaseModel):
    format: ObjectFormat
    size: int
    path: str


class PageInfoFull(BaseModel):
    has_previous_page: bool = Field(alias="hasPreviousPage")
    has_next_page: bool = Field(alias="hasNextPage")
    start_cursor: Optional[str] = Field(alias="startCursor")
    end_cursor: Optional[str] = Field(alias="endCursor")


class ProjectFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]


class ProteinConformerFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    residues: List[int]
    structure: "ProteinConformerFieldsStructure"
    protein: "ProteinConformerFieldsProtein"


class ProteinConformerFieldsStructure(BaseModel):
    id: Any
    rcsb_id: Optional[str]


class ProteinConformerFieldsProtein(BaseModel):
    id: Any
    sequence: str


class ProteinFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    sequence: str


class RunFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    rex: str
    result: Optional[Any]
    trace: Optional[Any]
    module_lock: Any
    status: RunStatus


class SmolConformerFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    residues: List[int]
    structure: "SmolConformerFieldsStructure"
    smol: "SmolConformerFieldsSmol"


class SmolConformerFieldsStructure(BaseModel):
    id: Any
    rcsb_id: Optional[str]


class SmolConformerFieldsSmol(BaseModel):
    id: Any
    smi: Optional[str]


class SmolFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    smi: Optional[str]
    inchi: Optional[str]
    data_blocks: Optional[List[List[str]]]


class SmolLibraryFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    size_hint: int


class SmolLibraryPartitionFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    part_idx: int
    part_size: int
    structures: Optional[Any]
    smiles: Any
    data_blocks: Any


class StructureFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    rcsb_id: Optional[str]
    topology: Any
    residues: Any
    chains: Any


class TagFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    key: str
    value: Optional[str]
    tagged_id: Any
    tagged_type: TaggedType


class TokenFields(BaseModel):
    id: Any
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]


class UserFields(BaseModel):
    id: UUID
    name: Optional[str]
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime]
    tags: Optional[List[str]]
    account: "UserFieldsAccount"


class UserFieldsAccount(BaseModel):
    id: UUID
    tier: AccountTier


AccountFields.model_rebuild()
ArgumentPartial.model_rebuild()
BenchmarkDataFields.model_rebuild()
BenchmarkFields.model_rebuild()
BenchmarkSubmissionFields.model_rebuild()
BindingAffinityFields.model_rebuild()
BindingPoseAffinityFields.model_rebuild()
BindingPoseConformerFields.model_rebuild()
BindingPoseConformerInteractionFields.model_rebuild()
BindingSiteConformerFields.model_rebuild()
BindingSiteInteractionFields.model_rebuild()
MSAFields.model_rebuild()
ModuleFull.model_rebuild()
ModuleInstanceBase.model_rebuild()
ModuleInstanceFull.model_rebuild()
ObjectFields.model_rebuild()
PageInfoFull.model_rebuild()
ProjectFields.model_rebuild()
ProteinConformerFields.model_rebuild()
ProteinFields.model_rebuild()
RunFields.model_rebuild()
SmolConformerFields.model_rebuild()
SmolFields.model_rebuild()
SmolLibraryFields.model_rebuild()
SmolLibraryPartitionFields.model_rebuild()
StructureFields.model_rebuild()
TagFields.model_rebuild()
TokenFields.model_rebuild()
UserFields.model_rebuild()
