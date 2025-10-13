"""
rush-py2: Python library for molecular structure file format conversion.
"""

from .mol import (
    Element, BondOrder, AtomRef, ResidueRef, ChainRef,
    FormalCharge, PartialCharge, Bond, Fragment,
    SchemaVersion, Topology, AminoAcidSeq, Residue, Residues,
    Chain, Chains, TRC, ResidueId
)

from .convert import (
    from_pdb, to_pdb,
    from_mmcif,
    from_json, to_json,
    load_structure, save_structure
)

__all__ = [
    # Data structures
    'Element', 'BondOrder', 'AtomRef', 'ResidueRef', 'ChainRef',
    'FormalCharge', 'PartialCharge', 'Bond', 'Fragment',
    'SchemaVersion', 'Topology', 'AminoAcidSeq', 'Residue', 'Residues',
    'Chain', 'Chains', 'TRC', 'ResidueId',
    
    # Conversion functions
    'from_pdb', 'to_pdb',
    'from_mmcif',
    'from_json', 'to_json',
    'load_structure', 'save_structure',
]

