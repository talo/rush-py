"""
Python library for accessing and writing computational chemistry workflows with
the Rush platform.
"""

from ._trc import TRCPaths, TRCRef
from .client import ObjectID, RunID, RushObject, RushRunError
from .convert import (
    from_json,
    from_mmcif,
    from_pdb,
    from_sdf,
    load_structure,
    merge_trcs,
    save_structure,
    to_json,
    to_pdb,
)
from .mol import (
    TRC,
    AminoAcidSeq,
    AtomRef,
    Bond,
    BondOrder,
    Chain,
    ChainRef,
    Chains,
    Element,
    FormalCharge,
    Fragment,
    PartialCharge,
    Residue,
    ResidueId,
    ResidueRef,
    Residues,
    SchemaVersion,
    Topology,
)
from .run import RushRun

__all__ = [
    # I/O
    "from_json",
    "to_json",
    "from_pdb",
    "to_pdb",
    "from_mmcif",
    "from_sdf",
    "load_structure",
    "save_structure",
    "merge_trcs",
    "TRCPaths",
    "TRCRef",
    "ObjectID",
    "RushObject",
    "RunID",
    "RushRunError",
    "RushRun",
    # Core structures
    "TRC",
    "Topology",
    "Residues",
    "Chains",
    # Chemistry types
    "Element",
    "Bond",
    "BondOrder",
    "FormalCharge",
    "PartialCharge",
    "Fragment",
    "AminoAcidSeq",
    "SchemaVersion",
    # Indices and records
    "AtomRef",
    "Residue",
    "ResidueRef",
    "ResidueId",
    "Chain",
    "ChainRef",
]
