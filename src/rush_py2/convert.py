"""
Python equivalent of the libqdx PDB structures and converter.

This module provides Python classes equivalent to the Rust structures:
- Topology, Residues, Chains, and TRC from libqdx
- PDB, mmCIF, and JSON parsing and writing functionality

The structures are designed to serialize to/from JSON exactly as the Rust libqdx implementation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Union, Any
from enum import Enum, IntEnum
import json
import re
from collections import defaultdict, OrderedDict


class Element(IntEnum):
    """Element enum equivalent to Rust Element enum."""
    X = 0
    H = 1
    He = 2
    Li = 3
    Be = 4
    B = 5
    C = 6
    N = 7
    O = 8
    F = 9
    Ne = 10
    Na = 11
    Mg = 12
    Al = 13
    Si = 14
    P = 15
    S = 16
    Cl = 17
    Ar = 18
    K = 19
    Ca = 20
    Sc = 21
    Ti = 22
    V = 23
    Cr = 24
    Mn = 25
    Fe = 26
    Co = 27
    Ni = 28
    Cu = 29
    Zn = 30
    Ga = 31
    Ge = 32
    As = 33
    Se = 34
    Br = 35
    Kr = 36
    
    @classmethod
    def from_str(cls, symbol: str) -> Element:
        """Parse element from string symbol."""
        # First try the symbol as-is (for proper case like "Fe")
        try:
            return cls[symbol]
        except KeyError:
            pass
        
        # Try uppercase (for "FE" -> "Fe")
        symbol_upper = symbol.upper()
        try:
            # Check all enum members for case-insensitive match
            for member in cls:
                if member.name.upper() == symbol_upper:
                    return member
        except:
            pass
        
        # Try common variations
        if symbol_upper in ['D']:  # Deuterium -> Hydrogen
            return cls.H
        
        raise ValueError(f"Unknown element symbol: {symbol}")
    
    def __str__(self) -> str:
        return self.name


class BondOrder(IntEnum):
    """Bond order enum."""
    Single = 1
    Double = 2
    Triple = 3
    OneAndAHalf = 4
    Ring = 5  # Aromatic


class AtomRef:
    """Reference to an atom by index. Equivalent to Rust AtomRef(u32).
    
    Rust tuple structs serialize to JSON as single values, not objects.
    AtomRef(5) becomes just 5 in JSON.
    """
    def __init__(self, value: int):
        if value < 0:
            raise ValueError("Atom index must be non-negative")
        self.value = value
    
    def __eq__(self, other):
        return isinstance(other, AtomRef) and self.value == other.value
    
    def __hash__(self):
        return hash(self.value)
    
    def __repr__(self):
        return f"AtomRef({self.value})"
    
    def __int__(self):
        return self.value


class ResidueRef:
    """Reference to a residue by index. Equivalent to Rust ResidueRef(u32).
    
    Rust tuple structs serialize to JSON as single values, not objects.
    ResidueRef(3) becomes just 3 in JSON.
    """
    def __init__(self, value: int):
        if value < 0:
            raise ValueError("Residue index must be non-negative")
        self.value = value
    
    def __eq__(self, other):
        return isinstance(other, ResidueRef) and self.value == other.value
    
    def __hash__(self):
        return hash(self.value)
    
    def __repr__(self):
        return f"ResidueRef({self.value})"
    
    def __int__(self):
        return self.value


class ChainRef:
    """Reference to a chain by index. Equivalent to Rust ChainRef(u32).
    
    Rust tuple structs serialize to JSON as single values, not objects.
    ChainRef(1) becomes just 1 in JSON.
    """
    def __init__(self, value: int):
        if value < 0:
            raise ValueError("Chain index must be non-negative")
        self.value = value
    
    def __eq__(self, other):
        return isinstance(other, ChainRef) and self.value == other.value
    
    def __hash__(self):
        return hash(self.value)
    
    def __repr__(self):
        return f"ChainRef({self.value})"
    
    def __int__(self):
        return self.value


@dataclass
class FormalCharge:
    """Formal charge of an atom."""
    charge: int


@dataclass
class PartialCharge:
    """Partial charge of an atom."""
    charge: float


@dataclass
class Bond:
    """Bond between two atoms."""
    atom1: AtomRef
    atom2: AtomRef
    order: BondOrder
    
    def __post_init__(self):
        if self.atom1.value == self.atom2.value:
            raise ValueError("Bond cannot connect an atom to itself")


class Fragment:
    """Fragment containing a list of atoms. Equivalent to Rust Fragment(Vec<AtomRef>).
    
    Rust tuple structs with Vec serialize to JSON as arrays.
    Fragment([AtomRef(1), AtomRef(2)]) becomes [1, 2] in JSON.
    """
    def __init__(self, atoms: List[AtomRef] = None):
        # Store as list of integers to match JSON serialization
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = [atom.value if isinstance(atom, AtomRef) else atom for atom in atoms]
    
    def __len__(self) -> int:
        return len(self.atoms)
    
    def __iter__(self):
        # Return AtomRef objects when iterating
        return (AtomRef(atom) for atom in self.atoms)
    
    def __eq__(self, other):
        return isinstance(other, Fragment) and self.atoms == other.atoms
    
    def __repr__(self):
        return f"Fragment({[AtomRef(a) for a in self.atoms]})"


class SchemaVersion(Enum):
    """Schema version for the topology format."""
    V1 = "v1"
    V2 = "v2"


@dataclass
class Topology:
    """
    Topology contains all atom information.
    
    This is equivalent to the Rust Topology struct.
    """
    schema_version: SchemaVersion = SchemaVersion.V2
    
    # Element of each atom
    symbols: List[Element] = field(default_factory=list)
    
    # XYZ coordinates of each atom (3 * len(symbols))
    geometry: List[float] = field(default_factory=list)
    
    # Optional atom labels
    labels: Optional[List[str]] = None
    
    # Optional partial charges
    partial_charges: Optional[List[PartialCharge]] = None
    
    # Optional formal charges
    formal_charges: Optional[List[FormalCharge]] = None
    
    # Optional connectivity
    connectivity: Optional[List[Bond]] = None
    
    # Optional velocities (3 * len(symbols))
    velocities: Optional[List[float]] = None
    
    # Optional fragments
    fragments: Optional[List[Fragment]] = None
    
    # Optional fragment charges
    fragment_formal_charges: Optional[List[FormalCharge]] = None
    fragment_partial_charges: Optional[List[PartialCharge]] = None
    
    def check(self) -> None:
        """Validate the topology structure."""
        # Check geometry length
        if len(self.geometry) != len(self.symbols) * 3:
            raise ValueError(f"Geometry length {len(self.geometry)} != symbols length {len(self.symbols)} * 3")
        
        # Check optional field lengths
        if self.labels is not None and len(self.labels) != len(self.symbols):
            raise ValueError(f"Labels length {len(self.labels)} != symbols length {len(self.symbols)}")
        
        if self.partial_charges is not None and len(self.partial_charges) != len(self.symbols):
            raise ValueError(f"Partial charges length {len(self.partial_charges)} != symbols length {len(self.symbols)}")
        
        if self.formal_charges is not None and len(self.formal_charges) != len(self.symbols):
            raise ValueError(f"Formal charges length {len(self.formal_charges)} != symbols length {len(self.symbols)}")
        
        if self.velocities is not None and len(self.velocities) != len(self.symbols) * 3:
            raise ValueError(f"Velocities length {len(self.velocities)} != symbols length {len(self.symbols)} * 3")
        
        # Check connectivity
        if self.connectivity is not None:
            for bond in self.connectivity:
                if bond.atom1.value >= len(self.symbols) or bond.atom2.value >= len(self.symbols):
                    raise ValueError(f"Bond references invalid atom indices: {bond.atom1.value}, {bond.atom2.value}")
        
        # Check fragments
        if self.fragments is not None:
            atom_set = set()
            for fragment in self.fragments:
                for atom_idx in fragment.atoms:
                    if atom_idx >= len(self.symbols):
                        raise ValueError(f"Fragment references invalid atom index: {atom_idx}")
                    if atom_idx in atom_set:
                        raise ValueError(f"Atom {atom_idx} appears in multiple fragments")
                    atom_set.add(atom_idx)
            
            if len(atom_set) != len(self.symbols):
                raise ValueError("Not all atoms are assigned to fragments")
    
    def distance_between_atoms(self, atom1: AtomRef, atom2: AtomRef) -> float:
        """Calculate distance between two atoms."""
        if atom1.value >= len(self.symbols) or atom2.value >= len(self.symbols):
            raise ValueError("Invalid atom indices")
        
        i1, i2 = atom1.value * 3, atom2.value * 3
        dx = self.geometry[i1] - self.geometry[i2]
        dy = self.geometry[i1 + 1] - self.geometry[i2 + 1]
        dz = self.geometry[i1 + 2] - self.geometry[i2 + 2]
        
        return (dx*dx + dy*dy + dz*dz)**0.5
    
    def distance_to_point(self, atom: AtomRef, point: Tuple[float, float, float]) -> float:
        """Calculate distance from atom to a point."""
        if atom.value >= len(self.symbols):
            raise ValueError("Invalid atom index")
        
        i = atom.value * 3
        dx = self.geometry[i] - point[0]
        dy = self.geometry[i + 1] - point[1]
        dz = self.geometry[i + 2] - point[2]
        
        return (dx*dx + dy*dy + dz*dz)**0.5
    
    def get_atoms_near_point(self, point: Tuple[float, float, float], 
                           threshold: float, atom_indices: Optional[List[int]] = None) -> List[int]:
        """Get atom indices within threshold distance of a point."""
        if atom_indices is None:
            atom_indices = list(range(len(self.symbols)))
        
        near_atoms = []
        for atom_idx in atom_indices:
            if atom_idx >= len(self.symbols):
                continue
            
            distance = self.distance_to_point(AtomRef(atom_idx), point)
            if distance <= threshold:
                near_atoms.append(atom_idx)
        
        return near_atoms
    
    def extend(self, other: 'Topology') -> None:
        """Extend this topology with atoms from another topology."""
        offset = len(self.symbols)
        
        # Extend basic arrays
        self.symbols.extend(other.symbols)
        self.geometry.extend(other.geometry)
        
        # Extend optional arrays
        if self.labels is not None and other.labels is not None:
            self.labels.extend(other.labels)
        elif self.labels is not None and other.labels is None:
            self.labels.extend([""] * len(other.symbols))
        
        if self.partial_charges is not None and other.partial_charges is not None:
            self.partial_charges.extend(other.partial_charges)
        elif self.partial_charges is not None and other.partial_charges is None:
            self.partial_charges.extend([PartialCharge(0.0)] * len(other.symbols))
        
        if self.formal_charges is not None and other.formal_charges is not None:
            self.formal_charges.extend(other.formal_charges)
        elif self.formal_charges is not None and other.formal_charges is None:
            self.formal_charges.extend([FormalCharge(0)] * len(other.symbols))
        
        if self.velocities is not None and other.velocities is not None:
            self.velocities.extend(other.velocities)
        elif self.velocities is not None and other.velocities is None:
            self.velocities.extend([0.0] * (len(other.symbols) * 3))
        
        # Update connectivity with offset
        if self.connectivity is not None and other.connectivity is not None:
            for bond in other.connectivity:
                new_bond = Bond(
                    AtomRef(bond.atom1.value + offset),
                    AtomRef(bond.atom2.value + offset),
                    bond.order
                )
                self.connectivity.append(new_bond)
        
        # Update fragments with offset
        if self.fragments is not None and other.fragments is not None:
            for fragment in other.fragments:
                new_atoms = [AtomRef(atom + offset) for atom in fragment.atoms]
                self.fragments.append(Fragment(new_atoms))
        elif self.fragments is not None and other.fragments is None:
            # Create a single fragment for all new atoms
            new_atoms = [AtomRef(i + offset) for i in range(len(other.symbols))]
            self.fragments.append(Fragment(new_atoms))
        
        # Extend fragment charges
        if self.fragment_formal_charges is not None and other.fragment_formal_charges is not None:
            self.fragment_formal_charges.extend(other.fragment_formal_charges)
        
        if self.fragment_partial_charges is not None and other.fragment_partial_charges is not None:
            self.fragment_partial_charges.extend(other.fragment_partial_charges)
    
    def new_topology_from_residue_subset(self, residue_subset: List[Residue]) -> 'Topology':
        """Create a new topology containing only atoms from specified residues."""
        new_topology = Topology(schema_version=self.schema_version)
        
        # Collect all atom indices from residues
        atom_indices = []
        for residue in residue_subset:
            atom_indices.extend(residue.atoms)  # Already integers
        
        # Build atom mapping
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(atom_indices)}
        
        # Copy basic data
        new_topology.symbols = [self.symbols[i] for i in atom_indices]
        new_topology.geometry = []
        for i in atom_indices:
            new_topology.geometry.extend(self.geometry[i*3:(i+1)*3])
        
        # Copy optional data
        if self.labels:
            new_topology.labels = [self.labels[i] for i in atom_indices]
        
        if self.partial_charges:
            new_topology.partial_charges = [self.partial_charges[i] for i in atom_indices]
        
        if self.formal_charges:
            new_topology.formal_charges = [self.formal_charges[i] for i in atom_indices]
        
        if self.velocities:
            new_topology.velocities = []
            for i in atom_indices:
                new_topology.velocities.extend(self.velocities[i*3:(i+1)*3])
        
        # Copy connectivity (only bonds between atoms in subset)
        if self.connectivity:
            new_topology.connectivity = []
            for bond in self.connectivity:
                if bond.atom1.value in old_to_new and bond.atom2.value in old_to_new:
                    new_bond = Bond(
                        AtomRef(old_to_new[bond.atom1.value]),
                        AtomRef(old_to_new[bond.atom2.value]),
                        bond.order
                    )
                    new_topology.connectivity.append(new_bond)
        
        return new_topology


class AminoAcidSeq(Enum):
    """Amino acid sequence names."""
    GLY = "GLY"
    ALA = "ALA"
    VAL = "VAL"
    LEU = "LEU"
    ILE = "ILE"
    PRO = "PRO"
    SER = "SER"
    THR = "THR"
    ASN = "ASN"
    GLN = "GLN"
    CYS = "CYS"
    CYD = "CYD"
    CYX = "CYX"
    MET = "MET"
    PHE = "PHE"
    TYR = "TYR"
    TYD = "TYD"
    TRP = "TRP"
    ASP = "ASP"
    ASH = "ASH"
    GLU = "GLU"
    GLH = "GLH"
    HIS = "HIS"
    HIN = "HIN"
    HID = "HID"
    HIE = "HIE"
    HIP = "HIP"
    LYS = "LYS"
    LYD = "LYD"
    LYN = "LYN"
    ARG = "ARG"
    HYP = "HYP"
    ACE = "ACE"
    BNC = "BNC"
    NME = "NME"
    NMA = "NMA"
    NHH = "NHH"
    UNK = "UNK"
    
    @classmethod
    def is_amino_acid(cls, residue_name: str) -> bool:
        """Check if a residue name is a known amino acid."""
        try:
            cls(residue_name.upper())
            return True
        except ValueError:
            return False
    
    def to_single_letter(self) -> str:
        """Convert to single letter code."""
        mapping = {
            'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I',
            'PRO': 'P', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q',
            'CYS': 'C', 'CYD': 'C', 'CYX': 'C', 'MET': 'M', 'PHE': 'F',
            'TYR': 'Y', 'TYD': 'Y', 'TRP': 'W', 'ASP': 'D', 'ASH': 'D',
            'GLU': 'E', 'GLH': 'E', 'HIS': 'H', 'HIN': 'H', 'HID': 'H',
            'HIE': 'H', 'HIP': 'H', 'LYS': 'K', 'LYD': 'K', 'LYN': 'K',
            'ARG': 'R', 'HYP': 'O'
        }
        return mapping.get(self.value, 'X')


class Residue:
    """A residue containing a list of atoms. Equivalent to Rust Residue(Vec<AtomRef>).
    
    Rust tuple structs with Vec serialize to JSON as arrays.
    Residue([AtomRef(1), AtomRef(2)]) becomes [1, 2] in JSON.
    """
    def __init__(self, atoms: List[AtomRef] = None):
        # Store as list of integers to match JSON serialization
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = [atom.value if isinstance(atom, AtomRef) else atom for atom in atoms]
    
    def __len__(self) -> int:
        return len(self.atoms)
    
    def __iter__(self):
        # Return AtomRef objects when iterating
        return (AtomRef(atom) for atom in self.atoms)
    
    def contains(self, atom: AtomRef) -> bool:
        return atom.value in self.atoms
    
    def __eq__(self, other):
        return isinstance(other, Residue) and self.atoms == other.atoms
    
    def __repr__(self):
        return f"Residue({[AtomRef(a) for a in self.atoms]})"


@dataclass
class Residues:
    """Collection of residues with metadata."""
    
    # List of residues
    residues: List[Residue] = field(default_factory=list)
    
    # Sequence names (e.g., amino acid names)
    seqs: List[str] = field(default_factory=list)
    
    # Sequence numbers
    seq_ns: List[int] = field(default_factory=list)
    
    # Insertion codes
    insertion_codes: List[str] = field(default_factory=list)
    
    def check(self) -> None:
        """Validate the residues structure."""
        if len(self.seqs) != len(self.residues):
            raise ValueError(f"Seqs length {len(self.seqs)} != residues length {len(self.residues)}")
        
        if len(self.seq_ns) != len(self.residues):
            raise ValueError(f"Seq_ns length {len(self.seq_ns)} != residues length {len(self.residues)}")
        
        if len(self.insertion_codes) != len(self.residues):
            raise ValueError(f"Insertion codes length {len(self.insertion_codes)} != residues length {len(self.residues)}")
    
    def is_amino_acid(self, index: int) -> bool:
        """Check if residue at index is an amino acid."""
        if index >= len(self.seqs):
            return False
        return AminoAcidSeq.is_amino_acid(self.seqs[index])
    
    def amino_acid_indices(self) -> List[int]:
        """Get indices of amino acid residues."""
        return [i for i in range(len(self.seqs)) if self.is_amino_acid(i)]
    
    def non_amino_acid_indices(self) -> List[int]:
        """Get indices of non-amino acid residues."""
        return [i for i in range(len(self.seqs)) if not self.is_amino_acid(i)]
    
    def extend(self, other: 'Residues') -> None:
        """Extend this residues collection with another."""
        # Calculate atom offset for renumbering
        offset = sum(len(residue.atoms) for residue in self.residues)
        
        # Extend residues with renumbered atoms
        for residue in other.residues:
            new_atoms = [atom + offset for atom in residue.atoms]
            self.residues.append(Residue(new_atoms))
        
        # Extend metadata
        self.seqs.extend(other.seqs)
        self.seq_ns.extend(other.seq_ns)
        self.insertion_codes.extend(other.insertion_codes)
    
    def new_residues_from_subset(self, residue_refs: List[ResidueRef]) -> 'Residues':
        """Create new residues collection from a subset of residue references."""
        new_residues = Residues()
        
        offset = 0
        for residue_ref in residue_refs:
            if residue_ref.value >= len(self.residues):
                continue
            
            # Get original residue
            original_residue = self.residues[residue_ref.value]
            residue_len = len(original_residue.atoms)
            
            # Create new residue with renumbered atoms
            new_atoms = [offset + i for i in range(residue_len)]
            new_residues.residues.append(Residue(new_atoms))
            
            # Copy metadata
            new_residues.seqs.append(self.seqs[residue_ref.value])
            new_residues.seq_ns.append(self.seq_ns[residue_ref.value])
            new_residues.insertion_codes.append(self.insertion_codes[residue_ref.value])
            
            offset += residue_len
        
        return new_residues


class Chain:
    """A chain containing a list of residues. Equivalent to Rust Chain(Vec<ResidueRef>).
    
    Rust tuple structs with Vec serialize to JSON as arrays.
    Chain([ResidueRef(1), ResidueRef(2)]) becomes [1, 2] in JSON.
    """
    def __init__(self, residues: List[ResidueRef] = None):
        # Store as list of integers to match JSON serialization
        if residues is None:
            self.residues = []
        else:
            self.residues = [res.value if isinstance(res, ResidueRef) else res for res in residues]
    
    def __len__(self) -> int:
        return len(self.residues)
    
    def __iter__(self):
        # Return ResidueRef objects when iterating
        return (ResidueRef(res) for res in self.residues)
    
    def contains(self, residue: ResidueRef) -> bool:
        return residue.value in self.residues
    
    def __eq__(self, other):
        return isinstance(other, Chain) and self.residues == other.residues
    
    def __repr__(self):
        return f"Chain({[ResidueRef(r) for r in self.residues]})"


@dataclass
class Chains:
    """Collection of chains with secondary structure information."""
    
    # List of chains
    chains: List[Chain] = field(default_factory=list)
    
    # Optional alpha helix residues
    alpha_helices: Optional[List[ResidueRef]] = None
    
    # Optional beta sheet residues
    beta_sheets: Optional[List[ResidueRef]] = None
    
    def check(self) -> None:
        """Validate the chains structure."""
        # Basic validation - more complex checks could be added
        pass
    
    def extend(self, other: 'Chains') -> None:
        """Extend this chains collection with another."""
        # Calculate residue offset
        residue_offset = sum(len(chain.residues) for chain in self.chains)
        
        # Extend chains with renumbered residue references
        for chain in other.chains:
            new_residue_refs = [ref + residue_offset for ref in chain.residues]
            self.chains.append(Chain(new_residue_refs))
        
        # Extend secondary structure info
        if self.alpha_helices is not None and other.alpha_helices is not None:
            new_alpha_helices = [ref.value + residue_offset for ref in other.alpha_helices]
            self.alpha_helices.extend([ResidueRef(ref) for ref in new_alpha_helices])
        
        if self.beta_sheets is not None and other.beta_sheets is not None:
            new_beta_sheets = [ref.value + residue_offset for ref in other.beta_sheets]
            self.beta_sheets.extend([ResidueRef(ref) for ref in new_beta_sheets])
    
    def new_chains_from_residue_subset(self, residue_refs: List[ResidueRef]) -> 'Chains':
        """Create new chains collection from a subset of residue references."""
        new_chains = Chains()
        
        # Create mapping from old residue indices to new ones
        old_to_new_residue = {ref.value: i for i, ref in enumerate(residue_refs)}
        
        # Group residues by their original chains
        chain_to_new_residues = defaultdict(list)
        
        for new_idx, residue_ref in enumerate(residue_refs):
            # Find which chain this residue belonged to
            for chain_idx, chain in enumerate(self.chains):
                if residue_ref.value in chain.residues:
                    chain_to_new_residues[chain_idx].append(new_idx)
                    break
        
        # Create new chains
        for chain_idx in sorted(chain_to_new_residues.keys()):
            new_chain_residues = chain_to_new_residues[chain_idx]
            # Sort by original sequence order
            original_chain = self.chains[chain_idx]
            new_chain_residues.sort(key=lambda new_idx: original_chain.residues.index(residue_refs[new_idx].value))
            new_chains.chains.append(Chain(new_chain_residues))
        
        # Filter secondary structure info
        if self.alpha_helices:
            new_alpha_helices = []
            for residue_ref in self.alpha_helices:
                if residue_ref.value in old_to_new_residue:
                    new_alpha_helices.append(ResidueRef(old_to_new_residue[residue_ref.value]))
            new_chains.alpha_helices = new_alpha_helices if new_alpha_helices else None
        
        if self.beta_sheets:
            new_beta_sheets = []
            for residue_ref in self.beta_sheets:
                if residue_ref.value in old_to_new_residue:
                    new_beta_sheets.append(ResidueRef(old_to_new_residue[residue_ref.value]))
            new_chains.beta_sheets = new_beta_sheets if new_beta_sheets else None
        
        return new_chains


@dataclass
class TRC:
    """
    Combined Topology, Residues, and Chains structure.
    
    This is the main wrapper class equivalent to the Rust TRC struct.
    """
    topology: Topology = field(default_factory=Topology)
    residues: Residues = field(default_factory=Residues)
    chains: Chains = field(default_factory=Chains)
    
    def check(self) -> None:
        """Validate the entire TRC structure."""
        self.topology.check()
        self.residues.check()
        self.chains.check()
        
        # Check that all atoms are in residues
        atom_set = set()
        for residue in self.residues.residues:
            for atom_idx in residue.atoms:
                if atom_idx in atom_set:
                    raise ValueError(f"Atom {atom_idx} appears in multiple residues")
                atom_set.add(atom_idx)
        
        if len(atom_set) != len(self.topology.symbols):
            raise ValueError("Not all atoms are assigned to residues")
        
        # Check that all residues are in chains
        residue_set = set()
        for chain in self.chains.chains:
            for residue_idx in chain.residues:
                if residue_idx >= len(self.residues.residues):
                    raise ValueError(f"Chain references invalid residue index: {residue_idx}")
                if residue_idx in residue_set:
                    raise ValueError(f"Residue {residue_idx} appears in multiple chains")
                residue_set.add(residue_idx)
        
        if len(residue_set) != len(self.residues.residues):
            raise ValueError("Not all residues are assigned to chains")
    
    def extend(self, other: 'TRC') -> None:
        """Extend this TRC with another TRC."""
        self.topology.extend(other.topology)
        self.residues.extend(other.residues)
        self.chains.extend(other.chains)
    
    def new_trc_from_residue_subset(self, residue_refs: List[ResidueRef]) -> 'TRC':
        """Create new TRC from a subset of residue references."""
        # Get residue subset
        residue_subset = [self.residues.residues[ref.value] for ref in residue_refs]
        
        return TRC(
            topology=self.topology.new_topology_from_residue_subset(residue_subset),
            residues=self.residues.new_residues_from_subset(residue_refs),
            chains=self.chains.new_chains_from_residue_subset(residue_refs)
        )


# PDB parsing structures and functions will be added in the next part
@dataclass
class PDBAtom:
    """Represents a parsed PDB ATOM/HETATM record."""
    atom_idx: int
    atom_name: str
    alternate_location: Optional[str]
    residue_name: str
    chain_id: str
    sequence_number: int
    residue_insertion: Optional[str]
    atom_x: float
    atom_y: float
    atom_z: float
    occupancy: float
    temperature_factor: float
    segment_id: Optional[str]
    element_symbol: Element
    charge: Optional[int]


@dataclass(frozen=True)
class ResidueId:
    """Unique identifier for a residue."""
    chain_id: str
    sequence_number: int
    insertion_code: str
    residue_name: str
    
    def __str__(self) -> str:
        return f"{self.chain_id}_{self.sequence_number:>9}_{self.insertion_code}_{self.residue_name}"


def _parse_pdb_atom_line(line: str, line_num: int) -> PDBAtom:
    """Parse a PDB ATOM or HETATM line."""
    if len(line) < 54:
        raise ValueError(f"Line {line_num}: ATOM/HETATM line too short")
    
    try:
        atom_idx = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        alternate_location = line[16].strip() if line[16].strip() else None
        residue_name = line[17:20].strip()
        chain_id = line[21].strip() if len(line) > 21 else ""
        sequence_number = int(line[22:26].strip()) if line[22:26].strip() else 1
        residue_insertion = line[26].strip() if len(line) > 26 and line[26].strip() else None
        
        atom_x = float(line[30:38].strip()) if line[30:38].strip() else 0.0
        atom_y = float(line[38:46].strip()) if line[38:46].strip() else 0.0
        atom_z = float(line[46:54].strip()) if line[46:54].strip() else 0.0
        
        occupancy = float(line[54:60].strip()) if len(line) > 60 and line[54:60].strip() else 1.0
        temperature_factor = float(line[60:66].strip()) if len(line) > 66 and line[60:66].strip() else 0.0
        
        segment_id = line[72:76].strip() if len(line) > 76 and line[72:76].strip() else None
        
        element_symbol_str = line[76:78].strip() if len(line) > 78 and line[76:78].strip() else atom_name[0]
        element_symbol = Element.from_str(element_symbol_str)
        
        charge = None
        if len(line) > 80 and line[78:80].strip():
            charge_str = line[78:80].strip()
            if charge_str:
                # Parse charge like "+1", "-2", etc.
                if charge_str[-1] in "+-":
                    sign = 1 if charge_str[-1] == "+" else -1
                    magnitude = int(charge_str[:-1]) if charge_str[:-1] else 1
                    charge = sign * magnitude
                else:
                    charge = int(charge_str)
        
        return PDBAtom(
            atom_idx=atom_idx,
            atom_name=atom_name,
            alternate_location=alternate_location,
            residue_name=residue_name,
            chain_id=chain_id,
            sequence_number=sequence_number,
            residue_insertion=residue_insertion,
            atom_x=atom_x,
            atom_y=atom_y,
            atom_z=atom_z,
            occupancy=occupancy,
            temperature_factor=temperature_factor,
            segment_id=segment_id,
            element_symbol=element_symbol,
            charge=charge
        )
    except (ValueError, IndexError) as e:
        raise ValueError(f"Line {line_num}: Error parsing ATOM/HETATM line: {e}")


def _parse_conect_line(line: str) -> List[int]:
    """Parse a CONECT line and return list of atom indices."""
    atom_idxs = []
    # CONECT format: positions 6-11, 11-16, 16-21, 21-26, 26-31 for atom indices
    start = 6
    while start < len(line):
        end = start + 5
        if end > len(line):
            end = len(line)
        atom_idx_str = line[start:end].strip()
        if atom_idx_str:
            try:
                atom_idxs.append(int(atom_idx_str))
            except ValueError:
                break
        else:
            break
        start = end
    return atom_idxs


def from_pdb(pdb_contents: str) -> List[TRC]:
    """
    Parse PDB file contents into TRC structures.
    
    Args:
        pdb_contents: String contents of a PDB file
        
    Returns:
        List of TRC structures (one per model in multi-model files)
    """
    trcs = []
    trc_atom_ids = []
    global_connectivity = []  # List of (origin, target, order) tuples
    
    lines = pdb_contents.strip().split('\n')
    line_iter = iter(enumerate(lines, 1))
    
    eof = False
    while not eof:
        # Storage for current model
        atoms = []
        atom_ids = []
        residue_data = OrderedDict()  # ResidueId -> atom indices
        chain_data = defaultdict(set)  # chain_id -> set of ResidueIds
        connectivity = []  # Local connectivity for this model
        
        in_model = False
        
        while True:
            try:
                line_num, line = next(line_iter)
            except StopIteration:
                eof = True
                break
            
            if len(line) < 6:
                continue
                
            record_type = line[:6].strip()
            
            if record_type == "MODEL":
                in_model = True
                
            elif record_type == "ENDMDL":
                in_model = False
                break
                
            elif record_type in ["ATOM", "HETATM"]:
                in_model = True
                
                try:
                    atom = _parse_pdb_atom_line(line, line_num)
                    
                    # Only process atoms with alternate location "A" or None
                    # Skip atoms with other alternate locations (e.g., "B", "C", etc.)
                    if atom.alternate_location is None or atom.alternate_location == "A":
                        atoms.append(atom)
                        atom_ids.append(atom.atom_idx)
                        
                        # Create residue identifier
                        # Note: insertion_code uses "~" for sorting (to sort after all letters)
                        # but the actual value stored in the residues structure is empty string
                        residue_id = ResidueId(
                            chain_id=atom.chain_id,
                            sequence_number=atom.sequence_number,
                            insertion_code=atom.residue_insertion or "~",
                            residue_name=atom.residue_name
                        )
                        
                        # Add to residue data
                        if residue_id not in residue_data:
                            residue_data[residue_id] = []
                        residue_data[residue_id].append(len(atoms) - 1)  # Index in atoms list
                        
                        # Add to chain data
                        chain_data[atom.chain_id].add(residue_id)
                    # else: skip atoms with other alternate locations
                    
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue
                    
            elif record_type == "CONECT":
                try:
                    atom_idxs = _parse_conect_line(line)
                    if len(atom_idxs) >= 2:
                        origin = atom_idxs[0]
                        for target in atom_idxs[1:]:
                            if in_model:
                                connectivity.append((origin, target, 1))
                            else:
                                global_connectivity.append((origin, target, 1))
                except (ValueError, IndexError):
                    continue
            
            elif record_type == "END":
                break
        
        # If no atoms were found, skip this model
        if not atoms:
            if eof:
                break
            else:
                continue
        
        # Build the TRC for this model
        trc = _build_trc(atoms, atom_ids, residue_data, chain_data, connectivity)
        trcs.append(trc)
        trc_atom_ids.append(atom_ids)
        
        if eof:
            break
    
    # Apply global connectivity to all models
    for trc, atom_ids in zip(trcs, trc_atom_ids):
        _apply_global_connectivity(trc, atom_ids, global_connectivity)
    
    # If no TRCs were created, return an empty one
    if not trcs:
        trcs.append(TRC())
    
    return trcs


def _build_trc(atoms: List[PDBAtom], atom_ids: List[int], 
               residue_data: OrderedDict, chain_data: Dict[str, Set[ResidueId]],
               connectivity: List[Tuple[int, int, int]]) -> TRC:
    """Build a TRC structure from parsed PDB data."""
    
    trc = TRC()
    
    # Build topology
    trc.topology.symbols = [atom.element_symbol for atom in atoms]
    trc.topology.geometry = []
    for atom in atoms:
        trc.topology.geometry.extend([atom.atom_x, atom.atom_y, atom.atom_z])
    
    trc.topology.labels = [atom.atom_name for atom in atoms]
    
    # Formal charges (per atom)
    atom_formal_charges = [atom.charge or 0 for atom in atoms]
    trc.topology.formal_charges = [FormalCharge(charge) for charge in atom_formal_charges]
    
    # Sort residues by ResidueId (chain_id, sequence_number, insertion_code, residue_name)
    # This matches the Rust BTreeMap ordering
    sorted_residue_ids = sorted(residue_data.keys(), 
                               key=lambda rid: (rid.chain_id, rid.sequence_number, 
                                              rid.insertion_code, rid.residue_name))
    
    # Build residues in sorted order
    residue_list = []
    seq_names = []
    seq_numbers = []
    insertion_codes_list = []
    
    for residue_id in sorted_residue_ids:
        atom_indices = residue_data[residue_id]
        residue_atoms = [AtomRef(idx) for idx in atom_indices]
        residue_list.append(Residue(residue_atoms))
        seq_names.append(residue_id.residue_name)
        seq_numbers.append(residue_id.sequence_number)
        # Convert "~" back to empty string for storage
        insertion_code = "" if residue_id.insertion_code == "~" else residue_id.insertion_code
        insertion_codes_list.append(insertion_code)
    
    trc.residues.residues = residue_list
    trc.residues.seqs = seq_names
    trc.residues.seq_ns = seq_numbers
    trc.residues.insertion_codes = insertion_codes_list
    
    # Build chains
    chains = []
    residue_id_to_index = {rid: idx for idx, rid in enumerate(sorted_residue_ids)}
    chain_ids = sorted(chain_data.keys())
    
    for chain_id in chain_ids:
        chain_residue_ids = chain_data[chain_id]
        # Sort residues in chain by sequence number
        sorted_residue_ids = sorted(chain_residue_ids, 
                                  key=lambda rid: (rid.sequence_number, rid.insertion_code))
        
        chain_residue_refs = [ResidueRef(residue_id_to_index[rid]) for rid in sorted_residue_ids]
        chains.append(Chain(chain_residue_refs))
    
    trc.chains.chains = chains
    trc.chains.labeled = [ChainRef(i) for i in range(len(chains))]
    trc.chains.labels = [[chain_id] for chain_id in chain_ids]
    
    # Create fragments (one per residue) - amino acids as default fragments
    trc.topology.fragments = [
        Fragment([AtomRef(atom_idx) for atom_idx in residue.atoms])
        for residue in trc.residues.residues
    ]
    
    # Process connectivity
    connectivity_deduper = {}  # (origin, target) -> order
    for origin_id, target_id, order in connectivity:
        # Convert atom IDs to indices
        try:
            origin_idx = atom_ids.index(origin_id)
        except ValueError:
            continue
        
        try:
            target_idx = atom_ids.index(target_id)
        except ValueError:
            continue
        
        # Check if reverse bond already exists (dedup)
        if (target_idx, origin_idx) in connectivity_deduper:
            continue
        
        # If same bond already exists, increment order (double bond)
        if (origin_idx, target_idx) in connectivity_deduper:
            connectivity_deduper[(origin_idx, target_idx)] += 1
        else:
            connectivity_deduper[(origin_idx, target_idx)] = order
    
    # Convert to Bond objects
    bonds = []
    for (origin_idx, target_idx), order in connectivity_deduper.items():
        bonds.append(Bond(
            AtomRef(min(origin_idx, target_idx)),
            AtomRef(max(origin_idx, target_idx)),
            BondOrder(order)
        ))
    trc.topology.connectivity = bonds
    
    # Calculate fragment formal charges (sum of atom charges in each residue)
    fragment_formal_charges = []
    for residue in trc.residues.residues:
        total_charge = sum(atom_formal_charges[atom_idx] for atom_idx in residue.atoms)
        fragment_formal_charges.append(FormalCharge(total_charge))
    trc.topology.fragment_formal_charges = fragment_formal_charges
    
    return trc


def _apply_global_connectivity(trc: TRC, atom_ids: List[int], 
                               global_connectivity: List[Tuple[int, int, int]]):
    """Apply global connectivity records to a TRC."""
    if not global_connectivity:
        return
    
    connectivity_deduper = {}  # (origin, target) -> order
    
    for origin_id, target_id, order in global_connectivity:
        # Convert atom IDs to indices
        try:
            origin_idx = atom_ids.index(origin_id)
        except ValueError:
            continue
        
        try:
            target_idx = atom_ids.index(target_id)
        except ValueError:
            continue
        
        # Check if reverse bond already exists (dedup)
        if (target_idx, origin_idx) in connectivity_deduper:
            continue
        
        # If same bond already exists, increment order (double bond)
        if (origin_idx, target_idx) in connectivity_deduper:
            connectivity_deduper[(origin_idx, target_idx)] += 1
        else:
            connectivity_deduper[(origin_idx, target_idx)] = order
    
    # Convert to Bond objects
    additional_bonds = []
    for (origin_idx, target_idx), order in connectivity_deduper.items():
        additional_bonds.append(Bond(
            AtomRef(min(origin_idx, target_idx)),
            AtomRef(max(origin_idx, target_idx)),
            BondOrder(order)
        ))
    
    # Add to existing connectivity
    if trc.topology.connectivity:
        trc.topology.connectivity.extend(additional_bonds)
    else:
        trc.topology.connectivity = additional_bonds


def to_pdb(trc: TRC) -> str:
    """
    Convert TRC structure to PDB format string.
    
    Args:
        trc: TRC structure to convert
        
    Returns:
        PDB format string
    """
    lines = []
    
    # Create mapping from residue to chain
    residue_to_chain = {}
    for chain_idx, chain in enumerate(trc.chains.chains):
        for residue_idx in chain.residues:
            residue_to_chain[residue_idx] = chain_idx
    
    atom_idx = 1
    for residue_idx, residue in enumerate(trc.residues.residues):
        chain_idx = residue_to_chain.get(residue_idx, 0)
        chain_id = chr(65 + chain_idx) if chain_idx < 26 else 'A'  # A, B, C, ...
        
        residue_name = trc.residues.seqs[residue_idx] if residue_idx < len(trc.residues.seqs) else "UNK"
        seq_num = trc.residues.seq_ns[residue_idx] if residue_idx < len(trc.residues.seq_ns) else 1
        insertion_code = trc.residues.insertion_codes[residue_idx] if residue_idx < len(trc.residues.insertion_codes) else ""
        
        for atom_idx in residue.atoms:
            if atom_idx >= len(trc.topology.symbols):
                continue
                
            element = trc.topology.symbols[atom_idx]
            atom_name = trc.topology.labels[atom_idx] if trc.topology.labels else str(element)
            
            x = trc.topology.geometry[atom_idx * 3] if atom_idx * 3 < len(trc.topology.geometry) else 0.0
            y = trc.topology.geometry[atom_idx * 3 + 1] if atom_idx * 3 + 1 < len(trc.topology.geometry) else 0.0
            z = trc.topology.geometry[atom_idx * 3 + 2] if atom_idx * 3 + 2 < len(trc.topology.geometry) else 0.0
            
            formal_charge = 0
            if trc.topology.formal_charges and atom_idx < len(trc.topology.formal_charges):
                formal_charge = trc.topology.formal_charges[atom_idx].charge
            
            # Format ATOM record
            record_type = "ATOM" if AminoAcidSeq.is_amino_acid(residue_name) else "HETATM"
            
            line = f"{record_type:<6}{atom_idx:>5} {atom_name:<4} {residue_name:>3} {chain_id}{seq_num:>4}{insertion_code:<1}   {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {str(element):>2}{formal_charge:+2d}"
            lines.append(line)
            atom_idx += 1
    
    lines.append("END")
    return '\n'.join(lines)


def from_json(json_content: str) -> List[TRC]:
    """
    Load TRC structures from JSON.
    
    Args:
        json_content: JSON string content
        
    Returns:
        List of TRC structures
    """
    data = json.loads(json_content)
    trcs = []
    
    for trc_data in data:
        # Load topology
        topology_data = trc_data["topology"]
        topology = Topology()
        topology.schema_version = SchemaVersion.V2  # Default, could parse from schema_version
        topology.symbols = [Element.from_str(s) for s in topology_data["symbols"]]
        topology.geometry = topology_data["geometry"]
        
        if "labels" in topology_data and topology_data["labels"]:
            topology.labels = topology_data["labels"]
        
        if "formal_charges" in topology_data and topology_data["formal_charges"]:
            topology.formal_charges = [FormalCharge(c) for c in topology_data["formal_charges"]]
        
        if "partial_charges" in topology_data and topology_data["partial_charges"]:
            topology.partial_charges = [PartialCharge(c) for c in topology_data["partial_charges"]]
        
        if "velocities" in topology_data and topology_data["velocities"]:
            topology.velocities = topology_data["velocities"]
        
        if "connectivity" in topology_data and topology_data["connectivity"]:
            # Connectivity format needs to be determined from actual JSON
            pass
        
        if "fragments" in topology_data and topology_data["fragments"]:
            topology.fragments = [Fragment(frag) for frag in topology_data["fragments"]]
        
        # Load residues
        residues_data = trc_data["residues"]
        residues = Residues()
        residues.residues = [Residue(res) for res in residues_data["residues"]]
        residues.seqs = residues_data["seqs"]
        residues.seq_ns = residues_data["seq_ns"]
        residues.insertion_codes = residues_data["insertion_codes"]
        
        # Load chains
        chains_data = trc_data["chains"]
        chains = Chains()
        chains.chains = [Chain(chain) for chain in chains_data["chains"]]
        
        if chains_data.get("alpha_helices"):
            chains.alpha_helices = [ResidueRef(r) for r in chains_data["alpha_helices"]]
        
        if chains_data.get("beta_sheets"):
            chains.beta_sheets = [ResidueRef(r) for r in chains_data["beta_sheets"]]
        
        # Create TRC
        trc = TRC(topology=topology, residues=residues, chains=chains)
        trcs.append(trc)
    
    return trcs


def to_json(trcs: List[TRC]) -> str:
    """
    Convert TRC structures to JSON.
    
    Args:
        trcs: List of TRC structures
        
    Returns:
        JSON string
    """
    data = []
    
    for trc in trcs:
        # Build topology dict with only the fields that exist in expected format
        topology_dict = {
            "schema_version": "0.2.0",
            "symbols": [str(symbol) for symbol in trc.topology.symbols],
            "geometry": trc.topology.geometry,
        }
        
        # Add optional fields only if they have data or are expected to be null
        if trc.topology.labels:
            topology_dict["labels"] = trc.topology.labels
        
        if trc.topology.formal_charges:
            topology_dict["formal_charges"] = [c.charge for c in trc.topology.formal_charges]
        
        # Always include connectivity and fragments as empty lists if not present (based on expected JSON)
        topology_dict["connectivity"] = []
        topology_dict["fragments"] = []
        topology_dict["fragment_formal_charges"] = []
        
        trc_data = {
            "topology": topology_dict,
            "residues": {
                "residues": [residue.atoms for residue in trc.residues.residues],
                "seqs": trc.residues.seqs,
                "seq_ns": trc.residues.seq_ns,
                "insertion_codes": trc.residues.insertion_codes,
                "labeled": None,
                "labels": None
            },
            "chains": {
                "chains": [chain.residues for chain in trc.chains.chains],
                "alpha_helices": [r.value for r in trc.chains.alpha_helices] if trc.chains.alpha_helices else None,
                "beta_sheets": [r.value for r in trc.chains.beta_sheets] if trc.chains.beta_sheets else None,
                "labeled": [r.value for r in trc.chains.labeled] if trc.chains.labeled else None,
                "labels": trc.chains.labels
            }
        }
        
        # Set connectivity if exists
        if trc.topology.connectivity:
            topology_dict["connectivity"] = [
                [bond.atom1.value, bond.atom2.value, bond.order.value] 
                for bond in trc.topology.connectivity
            ]
        
        # Set fragments if exists
        if trc.topology.fragments:
            topology_dict["fragments"] = [fragment.atoms for fragment in trc.topology.fragments]
        
        # Set fragment formal charges if exists
        if trc.topology.fragment_formal_charges:
            topology_dict["fragment_formal_charges"] = [c.charge for c in trc.topology.fragment_formal_charges]
        
        data.append(trc_data)
    
    return json.dumps(data, indent=2)


def _parse_mmcif_value(value: str) -> str:
    """Parse an mmCIF value, handling quoted strings and special characters."""
    value = value.strip()
    if value in ('.', '?'):
        return ''
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _parse_mmcif_loop(lines: List[str], start_idx: int, prefix: str) -> Tuple[Optional[Tuple[List[str], List[List[str]]]], int]:
    """
    Parse an mmCIF loop starting at start_idx.
    
    Returns:
        ((column_names, rows), next_idx) or (None, next_idx) if not a loop with the given prefix
    """
    i = start_idx
    
    # Check if this is a loop
    if i >= len(lines) or not lines[i].strip().startswith('loop_'):
        return (None, i)
    
    i += 1
    
    # Parse column names
    columns = []
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        if not line.startswith('_'):
            break
        if line.startswith(prefix):
            columns.append(line[len(prefix):])
        elif columns:  # Started collecting columns for this prefix, now hit a different prefix
            break
        i += 1
    
    if not columns:
        return (None, i)
    
    # Parse data rows (may span multiple lines)
    rows = []
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        if line.startswith('_') or line.startswith('loop_'):
            break
        
        # Parse fields from current line (and additional lines if needed)
        fields = []
        current_line = lines[i]
        i += 1
        
        while len(fields) < len(columns):
            # Parse tokens from current_line
            tokens = []
            j = 0
            current_line_stripped = current_line.rstrip('\n\r')
            while j < len(current_line_stripped):
                # Skip whitespace
                while j < len(current_line_stripped) and current_line_stripped[j] in ' \t':
                    j += 1
                if j >= len(current_line_stripped):
                    break
                
                # Check for quoted string
                if current_line_stripped[j] in ("'", '"'):
                    quote_char = current_line_stripped[j]
                    j += 1
                    start = j
                    while j < len(current_line_stripped) and current_line_stripped[j] != quote_char:
                        j += 1
                    tokens.append(current_line_stripped[start:j])
                    j += 1  # Skip closing quote
                else:
                    # Unquoted value
                    start = j
                    while j < len(current_line_stripped) and current_line_stripped[j] not in ' \t':
                        j += 1
                    tokens.append(current_line_stripped[start:j])
            
            fields.extend(tokens)
            
            # If we don't have enough fields yet, try to read the next line
            if len(fields) < len(columns):
                if i < len(lines):
                    next_line = lines[i].strip()
                    if (next_line and not next_line.startswith('_') and 
                        not next_line.startswith('loop_') and 
                        not next_line.startswith('data_')):
                        current_line = lines[i]
                        i += 1
                    else:
                        break
                else:
                    break
        
        if len(fields) == len(columns):
            rows.append(fields)
    
    return ((columns, rows), i)


def from_mmcif(mmcif_content: str) -> List[TRC]:
    """
    Parse mmCIF file contents into TRC structures.
    
    Args:
        mmcif_content: String contents of an mmCIF file
        
    Returns:
        List of TRC structures
    """
    lines = mmcif_content.split('\n')
    trcs = []
    
    # Parse loops
    models = defaultdict(list)  # model_num -> list of atoms
    atom_loop_data = None
    struct_conn_data = None
    comp_bond_data = None
    
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('loop_'):
            # Try to parse atom_site loop
            result, next_i = _parse_mmcif_loop(lines, i, '_atom_site.')
            if result:
                columns, rows = result
                # Check if this has atom_site columns
                if any('id' in col or 'type_symbol' in col for col in columns):
                    atom_loop_data = (columns, rows)
                i = next_i
                continue
            
            # Try to parse struct_conn loop
            result, next_i = _parse_mmcif_loop(lines, i, '_struct_conn.')
            if result:
                struct_conn_data = result
                i = next_i
                continue
            
            # Try to parse chem_comp_bond loop
            result, next_i = _parse_mmcif_loop(lines, i, '_chem_comp_bond.')
            if result:
                comp_bond_data = result
                i = next_i
                continue
            
            i = next_i
        else:
            i += 1
    
    if not atom_loop_data:
        empty_trc = TRC()
        empty_trc.chains.labeled = []
        empty_trc.chains.labels = []
        return [empty_trc]
    
    columns, rows = atom_loop_data
    
    # Find column indices
    col_idx = {}
    for idx, col in enumerate(columns):
        col_idx[col] = idx
    
    # Parse atoms
    for row in rows:
        if len(row) < len(columns):
            continue
        
        def get_val(name: str, default: str = '') -> str:
            idx = col_idx.get(name)
            if idx is not None and idx < len(row):
                val = _parse_mmcif_value(row[idx])
                return val if val else default
            return default
        
        def get_int(name: str, default: int = 0) -> Optional[int]:
            val = get_val(name)
            if not val:
                return None
            try:
                return int(val)
            except ValueError:
                return None
        
        def get_int_with_default(name: str, default: int = 0) -> int:
            val = get_int(name)
            return val if val is not None else default
        
        def get_float(name: str, default: float = 0.0) -> float:
            val = get_val(name)
            try:
                return float(val) if val else default
            except ValueError:
                return default
        
        # Parse auth_seq_id with fallback logic matching Rust
        auth_seq_id_val = get_int('auth_seq_id')
        if auth_seq_id_val is None:
            auth_seq_id_val = get_int('label_seq_id')
            if auth_seq_id_val is None:
                auth_seq_id_val = 0
        
        atom = {
            'id': get_int_with_default('id', 0),
            'type_symbol': get_val('type_symbol', 'C'),
            'label_atom_id': get_val('label_atom_id', 'C'),
            'label_alt_id': get_val('label_alt_id', ''),
            'label_comp_id': get_val('label_comp_id', 'UNK'),
            'label_asym_id': get_val('label_asym_id', 'A'),
            'label_seq_id': get_int_with_default('label_seq_id', 0),
            'pdbx_PDB_ins_code': get_val('pdbx_PDB_ins_code', ''),
            'Cartn_x': get_float('Cartn_x', 0.0),
            'Cartn_y': get_float('Cartn_y', 0.0),
            'Cartn_z': get_float('Cartn_z', 0.0),
            'occupancy': get_float('occupancy', 1.0),
            'B_iso_or_equiv': get_float('B_iso_or_equiv', 0.0),
            'pdbx_formal_charge': get_int_with_default('pdbx_formal_charge', 0),
            'auth_asym_id': get_val('auth_asym_id', '') or get_val('label_asym_id', 'A'),
            'auth_seq_id': auth_seq_id_val,
            'group_PDB': get_val('group_PDB', 'ATOM'),
            'pdbx_PDB_model_num': get_val('pdbx_PDB_model_num', '1'),
        }
        
        model_num = atom['pdbx_PDB_model_num']
        models[model_num].append(atom)
    
    # Build TRC for each model
    for model_num in sorted(models.keys()):
        atoms = models[model_num]
        trc = _build_trc_from_mmcif_atoms(atoms, struct_conn_data, comp_bond_data)
        trcs.append(trc)
    
    if not trcs:
        empty_trc = TRC()
        empty_trc.chains.labeled = []
        empty_trc.chains.labels = []
        trcs.append(empty_trc)
    
    return trcs


def _build_trc_from_mmcif_atoms(atoms: List[Dict], 
                                struct_conn_data: Optional[Tuple[List[str], List[List[str]]]], 
                                comp_bond_data: Optional[Tuple[List[str], List[List[str]]]]) -> TRC:
    """Build a TRC from parsed mmCIF atoms."""
    trc = TRC()
    
    atom_ids = []
    atom_labels = []
    atom_formal_charges = []
    atom_symbols = []
    geometry = []
    
    residue_data = OrderedDict()
    chain_data = defaultdict(set)
    atom_index_map = {}  # Original atom index to topology index
    
    for orig_idx, atom in enumerate(atoms):
        # Only process atoms with alternate location "A" or None
        alt_id = atom['label_alt_id']
        if alt_id and alt_id != 'A':
            continue
        
        # Parse element from type_symbol
        type_symbol = atom['type_symbol']
        # Remove non-alphabetic characters
        element_str = ''.join(c for c in type_symbol if c.isalpha())
        try:
            element = Element.from_str(element_str)
        except (ValueError, KeyError):
            element = Element.C  # Default to carbon
        
        topology_idx = len(atom_symbols)
        atom_index_map[orig_idx] = topology_idx
        
        atom_symbols.append(element)
        geometry.extend([atom['Cartn_x'], atom['Cartn_y'], atom['Cartn_z']])
        
        atom_ids.append(atom['id'])
        atom_labels.append(atom['label_atom_id'])
        atom_formal_charges.append(atom['pdbx_formal_charge'])
        
        # Create residue identifier using auth fields and "~" for sorting
        residue_id = ResidueId(
            chain_id=atom['auth_asym_id'],
            sequence_number=atom['auth_seq_id'],
            insertion_code=atom['pdbx_PDB_ins_code'] or "~",
            residue_name=atom['label_comp_id']
        )
        
        if residue_id not in residue_data:
            residue_data[residue_id] = []
        residue_data[residue_id].append(len(atom_symbols) - 1)
        
        chain_data[atom['auth_asym_id']].add(residue_id)
    
    # Build topology
    trc.topology.symbols = atom_symbols
    trc.topology.geometry = geometry
    trc.topology.labels = atom_labels
    trc.topology.formal_charges = [FormalCharge(charge) for charge in atom_formal_charges]
    
    # Sort residues by ResidueId
    sorted_residue_ids = sorted(residue_data.keys(),
                               key=lambda rid: (rid.chain_id, rid.sequence_number,
                                              rid.insertion_code, rid.residue_name))
    
    # Build residues
    residue_list = []
    seq_names = []
    seq_numbers = []
    insertion_codes_list = []
    
    for residue_id in sorted_residue_ids:
        atom_indices = residue_data[residue_id]
        residue_list.append(Residue([AtomRef(idx) for idx in atom_indices]))
        seq_names.append(residue_id.residue_name)
        seq_numbers.append(residue_id.sequence_number)
        # Convert "~" back to empty string
        insertion_code = "" if residue_id.insertion_code == "~" else residue_id.insertion_code
        insertion_codes_list.append(insertion_code)
    
    trc.residues.residues = residue_list
    trc.residues.seqs = seq_names
    trc.residues.seq_ns = seq_numbers
    trc.residues.insertion_codes = insertion_codes_list
    
    # Build chains
    chains = []
    residue_id_to_index = {rid: idx for idx, rid in enumerate(sorted_residue_ids)}
    chain_ids = sorted(chain_data.keys())
    
    for chain_id in chain_ids:
        chain_residue_ids = chain_data[chain_id]
        sorted_chain_residue_ids = sorted(chain_residue_ids,
                                         key=lambda rid: (rid.sequence_number,
                                                         rid.insertion_code,
                                                         rid.residue_name))
        
        chain_residue_refs = [ResidueRef(residue_id_to_index[rid]) 
                             for rid in sorted_chain_residue_ids
                             if rid in residue_id_to_index]
        chains.append(Chain(chain_residue_refs))
    
    trc.chains.chains = chains
    trc.chains.labeled = [ChainRef(i) for i in range(len(chains))]
    trc.chains.labels = [[chain_id] for chain_id in chain_ids]
    
    # Create fragments (one per residue)
    trc.topology.fragments = [
        Fragment([AtomRef(atom_idx) for atom_idx in residue.atoms])
        for residue in trc.residues.residues
    ]
    
    # Calculate fragment formal charges
    fragment_formal_charges = []
    for residue in trc.residues.residues:
        total_charge = sum(atom_formal_charges[atom_idx] for atom_idx in residue.atoms)
        fragment_formal_charges.append(FormalCharge(total_charge))
    trc.topology.fragment_formal_charges = fragment_formal_charges
    
    # Build connectivity from struct_conn and chem_comp_bond
    connectivity_deduper = {}  # (min_idx, max_idx) -> bond_order
    
    # Parse struct_conn (inter-residue bonds)
    if struct_conn_data:
        columns, rows = struct_conn_data
        col_idx = {col: idx for idx, col in enumerate(columns)}
        
        for row in rows:
            def get_val(name: str) -> str:
                idx = col_idx.get(name)
                if idx is not None and idx < len(row):
                    return _parse_mmcif_value(row[idx])
                return ''
            
            def get_int_val(name: str) -> int:
                val = get_val(name)
                try:
                    return int(val) if val else 0
                except ValueError:
                    return 0
            
            # Find atoms by label (uses label_ fields, not auth_)
            ptnr1_atom = get_val('ptnr1_label_atom_id')
            ptnr1_asym = get_val('ptnr1_label_asym_id')  
            ptnr1_seq = get_int_val('ptnr1_label_seq_id')
            ptnr2_atom = get_val('ptnr2_label_atom_id')
            ptnr2_asym = get_val('ptnr2_label_asym_id')
            ptnr2_seq = get_int_val('ptnr2_label_seq_id')
            conn_type = get_val('conn_type_id')
            
            # Find matching atoms using label_ fields (find FIRST match like Rust .position())
            atom1_orig_idx = None
            atom2_orig_idx = None
            for idx, atom in enumerate(atoms):
                if atom1_orig_idx is None and (atom['label_atom_id'] == ptnr1_atom and 
                    atom['label_asym_id'] == ptnr1_asym and 
                    atom['label_seq_id'] == ptnr1_seq):
                    atom1_orig_idx = idx
                if atom2_orig_idx is None and (atom['label_atom_id'] == ptnr2_atom and 
                    atom['label_asym_id'] == ptnr2_asym and 
                    atom['label_seq_id'] == ptnr2_seq):
                    atom2_orig_idx = idx
                if atom1_orig_idx is not None and atom2_orig_idx is not None:
                    break
            
            if atom1_orig_idx is not None and atom2_orig_idx is not None:
                topo_idx1 = atom_index_map.get(atom1_orig_idx)
                topo_idx2 = atom_index_map.get(atom2_orig_idx)
                
                if topo_idx1 is not None and topo_idx2 is not None:
                    bond_order = 1  # Default to single bond
                    if conn_type in ['covale', 'metalc', 'disulf']:
                        bond_order = 1
                    
                    min_idx = min(topo_idx1, topo_idx2)
                    max_idx = max(topo_idx1, topo_idx2)
                    connectivity_deduper[(min_idx, max_idx)] = bond_order
    
    # Parse chem_comp_bond (intra-residue bonds)
    if comp_bond_data:
        columns, rows = comp_bond_data
        col_idx = {col: idx for idx, col in enumerate(columns)}
        
        # Build mapping of comp_id -> bonds
        comp_bonds = defaultdict(list)
        for row in rows:
            def get_val(name: str) -> str:
                idx = col_idx.get(name)
                if idx is not None and idx < len(row):
                    return _parse_mmcif_value(row[idx])
                return ''
            
            comp_id = get_val('comp_id')
            atom_id_1 = get_val('atom_id_1')
            atom_id_2 = get_val('atom_id_2')
            value_order = get_val('value_order')
            
            comp_bonds[comp_id].append((atom_id_1, atom_id_2, value_order))
        
        # Group atoms by residue for efficient lookup
        # Note: Rust uses (comp_id, auth_asym_id, auth_seq_id) without insertion code
        residue_atoms = defaultdict(list)  # (comp_id, auth_asym_id, auth_seq_id) -> list of (orig_idx, topo_idx, atom)
        for orig_idx, atom in enumerate(atoms):
            if (atom['label_alt_id'] == '' or atom['label_alt_id'] == 'A'):
                topo_idx = atom_index_map.get(orig_idx)
                if topo_idx is not None:
                    key = (atom['label_comp_id'], atom['auth_asym_id'], atom['auth_seq_id'])
                    residue_atoms[key].append((orig_idx, topo_idx, atom))
        
        # Apply bond definitions to residues
        for (comp_id, chain_id, seq_id), res_atoms in residue_atoms.items():
            if comp_id in comp_bonds:
                for atom_id_1, atom_id_2, value_order in comp_bonds[comp_id]:
                    # Find THE FIRST atom that matches each atom_id (Rust uses find())
                    topo_idx1 = None
                    topo_idx2 = None
                    for _, topo_idx, atom in res_atoms:
                        if topo_idx1 is None and atom['label_atom_id'] == atom_id_1:
                            topo_idx1 = topo_idx
                        if topo_idx2 is None and atom['label_atom_id'] == atom_id_2:
                            topo_idx2 = topo_idx
                        if topo_idx1 is not None and topo_idx2 is not None:
                            break
                    
                    if topo_idx1 is not None and topo_idx2 is not None:
                        # Parse bond order
                        bond_order = 1
                        if value_order == 'SING':
                            bond_order = 1
                        elif value_order == 'DOUB':
                            bond_order = 2
                        elif value_order == 'TRIP':
                            bond_order = 3
                        elif value_order == 'QUAD':
                            bond_order = 4
                        elif value_order == 'AROM':
                            bond_order = 5
                        
                        min_idx = min(topo_idx1, topo_idx2)
                        max_idx = max(topo_idx1, topo_idx2)
                        connectivity_deduper[(min_idx, max_idx)] = bond_order
    
    # Convert to Bond objects
    bonds = []
    for (min_idx, max_idx), order in sorted(connectivity_deduper.items()):
        bonds.append(Bond(
            AtomRef(min_idx),
            AtomRef(max_idx),
            BondOrder(order)
        ))
    trc.topology.connectivity = bonds
    
    return trc


def load_structure(file_path: str) -> List[TRC]:
    """
    Load structure from PDB, mmCIF, or JSON file.
    
    Args:
        file_path: Path to structure file
        
    Returns:
        List of TRC structures
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Determine file type by extension
    if file_path.lower().endswith('.json'):
        return from_json(content)
    elif file_path.lower().endswith('.cif'):
        return from_mmcif(content)
    elif file_path.lower().endswith('.pdb'):
        return from_pdb(content)
    else:
        # Try to guess from content
        content_lower = content.lower()
        if content.strip().startswith('[') or content.strip().startswith('{'):
            return from_json(content)
        elif 'data_' in content_lower and '_atom_site' in content_lower:
            return from_mmcif(content)
        else:
            return from_pdb(content)


def save_structure(trcs: List[TRC], file_path: str, format: str = None):
    """
    Save TRC structures to file.
    
    Args:
        trcs: List of TRC structures
        file_path: Output file path
        format: Output format ('pdb', 'json', or None for auto-detect from extension)
    """
    if format is None:
        # Auto-detect from extension
        if file_path.lower().endswith('.json'):
            format = 'json'
        elif file_path.lower().endswith('.pdb'):
            format = 'pdb'
        else:
            format = 'pdb'  # Default
    
    if format.lower() == 'json':
        content = to_json(trcs)
    elif format.lower() == 'pdb':
        if len(trcs) > 1:
            # Multi-model PDB
            content_parts = []
            for i, trc in enumerate(trcs, 1):
                content_parts.append(f"MODEL     {i:>4}")
                content_parts.append(to_pdb(trc).replace("END\n", ""))
                content_parts.append("ENDMDL")
            content_parts.append("END")
            content = '\n'.join(content_parts)
        else:
            content = to_pdb(trcs[0])
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    with open(file_path, 'w') as f:
        f.write(content)
