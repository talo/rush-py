"""
Provides data structures and helpers for molecular systems and structures:

- Classes Rush Topology, Residues, Chains, and TRC types.
- Element types and bonds.
- Fragment type to represent fragmented systems.

Quick Links
-----------

- :class:`rush.mol.TRC`
- :class:`rush.mol.Topology`
- :class:`rush.mol.Residues`
- :class:`rush.mol.Chains`
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Self


class Element(IntEnum):
    """Represents all relevant elements."""

    X = 0
    H = 1
    He = 2
    Li = 3
    Be = 4
    B = 5
    C = 6
    N = 7
    O = 8  # noqa: E741
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
    def from_str(cls, symbol: str) -> Self:
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
        except Exception:
            pass

        # Try common variations
        if symbol_upper in ["D"]:  # Deuterium -> Hydrogen
            return cls.H

        raise ValueError(f"Unknown element symbol: {symbol}")

    def __str__(self) -> str:
        return self.name


class AtomRef:
    """Reference to an atom by index."""

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


class FragmentRef:
    """Reference to a fragment by index."""

    def __init__(self, value: int):
        if value < 0:
            raise ValueError("Fragment index must be non-negative")
        self.value = value

    def __eq__(self, other):
        return isinstance(other, FragmentRef) and self.value == other.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"FragmentRef({self.value})"

    def __int__(self):
        return self.value


class ResidueRef:
    """Reference to a residue by index."""

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
    """Reference to a chain by index."""

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

    def __repr__(self):
        return f"FormalCharge({self.charge})"

    def __int__(self):
        return self.charge


@dataclass
class PartialCharge:
    """Partial charge of an atom."""

    charge: float

    def __repr__(self):
        return f"PartialCharge({self.charge})"

    def __float__(self):
        return self.charge


class BondOrder(IntEnum):
    """Bond order enum."""

    Single = 1
    Double = 2
    Triple = 3
    OneAndAHalf = 4  # Partial bond (e.g. amide bond)
    Ring = 5  # Aromatic


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
    """Fragment containing a list of atoms."""

    def __init__(self, atoms: list[AtomRef] | list[int] | None = None):
        # Store as list of integers to match JSON serialization
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = [
                atom.value if isinstance(atom, AtomRef) else atom for atom in atoms
            ]

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
    """Topology contains all atom information."""

    schema_version: SchemaVersion = SchemaVersion.V2

    # Element of each atom
    symbols: list[Element] = field(default_factory=list)

    # XYZ coordinates of each atom (3 * len(symbols))
    geometry: list[float] = field(default_factory=list)

    # Optional atom labels
    labels: list[str] | None = None

    # Optional partial charges
    partial_charges: list[PartialCharge] | None = None

    # Optional formal charges
    formal_charges: list[FormalCharge] | None = None

    # Optional connectivity
    connectivity: list[Bond] | None = None

    # Optional velocities (3 * len(symbols))
    velocities: list[float] | None = None

    # Optional fragments
    fragments: list[Fragment] | None = None

    # Optional fragment charges
    fragment_formal_charges: list[FormalCharge] | None = None
    fragment_partial_charges: list[PartialCharge] | None = None

    @staticmethod
    def from_json(json_content: str | Path | dict) -> "Topology":
        if isinstance(json_content, str):
            topology_data = json.loads(json_content)
        elif isinstance(json_content, Path):
            with json_content.open() as f:
                topology_data = json.load(f)
        elif isinstance(json_content, dict):
            topology_data = json_content
        else:
            print(
                "WARNING: Tried to load Topology from JSON but "
                "it wasn't a str, Path, or dict!"
            )
            topology_data = json_content

        topology = Topology()

        # Default, could parse from schema_version
        topology.schema_version = SchemaVersion.V2

        topology.symbols = [Element.from_str(s) for s in topology_data["symbols"]]
        topology.geometry = topology_data["geometry"]

        if "labels" in topology_data and topology_data["labels"]:
            topology.labels = topology_data["labels"]

        if "formal_charges" in topology_data and topology_data["formal_charges"]:
            topology.formal_charges = [
                FormalCharge(c) for c in topology_data["formal_charges"]
            ]

        if "partial_charges" in topology_data and topology_data["partial_charges"]:
            topology.partial_charges = [
                PartialCharge(c) for c in topology_data["partial_charges"]
            ]

        if "velocities" in topology_data and topology_data["velocities"]:
            topology.velocities = topology_data["velocities"]

        if "connectivity" in topology_data and topology_data["connectivity"]:
            # Connectivity is a list of [atom1, atom2, bond_order]
            # BondOrder enum: 1=Single, 2=Double, 3=Triple, 4=OneAndAHalf (partial), 5=Ring (aromatic)
            bonds = []
            for bond_data in topology_data["connectivity"]:
                if isinstance(bond_data, list) and len(bond_data) >= 2:
                    atom1_idx = bond_data[0]
                    atom2_idx = bond_data[1]
                    bond_order_val = bond_data[2]

                    # Support old version mapping: 254 -> 4 (OneAndAHalf/partial), 255 -> 5 (Ring/aromatic)
                    if bond_order_val == 254:
                        bond_order_val = 4
                    elif bond_order_val == 255:
                        bond_order_val = 5

                    bond_order = BondOrder(bond_order_val)
                    bonds.append(
                        Bond(AtomRef(atom1_idx), AtomRef(atom2_idx), bond_order)
                    )
            topology.connectivity = bonds

        if "fragments" in topology_data and topology_data["fragments"]:
            topology.fragments = [Fragment(frag) for frag in topology_data["fragments"]]

        if (
            "fragment_formal_charges" in topology_data
            and topology_data["fragment_formal_charges"]
        ):
            topology.fragment_formal_charges = [
                FormalCharge(c) for c in topology_data["fragment_formal_charges"]
            ]

        if (
            "fragment_partial_charges" in topology_data
            and topology_data["fragment_partial_charges"]
        ):
            topology.fragment_partial_charges = [
                PartialCharge(c) for c in topology_data["fragment_partial_charges"]
            ]

        return topology

    def check(self) -> None:
        """Validate the topology structure."""
        # Check geometry length
        if len(self.geometry) != len(self.symbols) * 3:
            raise ValueError(
                f"Geometry length {len(self.geometry)} != symbols length {len(self.symbols)} * 3"
            )

        # Check optional field lengths
        if self.labels is not None and len(self.labels) != len(self.symbols):
            raise ValueError(
                f"Labels length {len(self.labels)} != symbols length {len(self.symbols)}"
            )

        if self.partial_charges is not None and len(self.partial_charges) != len(
            self.symbols
        ):
            raise ValueError(
                f"Partial charges length {len(self.partial_charges)} != symbols length {len(self.symbols)}"
            )

        if self.formal_charges is not None and len(self.formal_charges) != len(
            self.symbols
        ):
            raise ValueError(
                f"Formal charges length {len(self.formal_charges)} != symbols length {len(self.symbols)}"
            )

        if (
            self.velocities is not None
            and len(self.velocities) != len(self.symbols) * 3
        ):
            raise ValueError(
                f"Velocities length {len(self.velocities)} != symbols length {len(self.symbols)} * 3"
            )

        # Check connectivity
        if self.connectivity is not None:
            for bond in self.connectivity:
                if bond.atom1.value >= len(self.symbols) or bond.atom2.value >= len(
                    self.symbols
                ):
                    raise ValueError(
                        f"Bond references invalid atom indices: {bond.atom1.value}, {bond.atom2.value}"
                    )

        # Check fragments
        if self.fragments is not None:
            atom_set = set()
            for fragment in self.fragments:
                for atom_idx in fragment.atoms:
                    if atom_idx >= len(self.symbols):
                        raise ValueError(
                            f"Fragment references invalid atom index: {atom_idx}"
                        )
                    if atom_idx in atom_set:
                        raise ValueError(
                            f"Atom {atom_idx} appears in multiple fragments"
                        )
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

        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def distance_to_point(
        self, atom: AtomRef, point: tuple[float, float, float]
    ) -> float:
        """Calculate distance from atom to a point."""
        if atom.value >= len(self.symbols):
            raise ValueError("Invalid atom index")

        i = atom.value * 3
        dx = self.geometry[i] - point[0]
        dy = self.geometry[i + 1] - point[1]
        dz = self.geometry[i + 2] - point[2]

        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def get_atoms_near_point(
        self,
        point: tuple[float, float, float],
        threshold: float,
        atom_indices: list[int] | None = None,
    ) -> list[int]:
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

    def get_fragments_near_fragment(
        self,
        frag_idx: int,
        threshold: float,
        atom_indices: list[int] | None = None,
    ) -> list[FragmentRef]:
        """Get fragment indices within threshold distance of another fragment."""
        if not self.fragments:
            return []

        if atom_indices is None:
            atom_indices = list(range(len(self.symbols)))

        near_atoms = set()
        for atom_idx in self.fragments[frag_idx]:
            atom_idx = int(atom_idx)
            if atom_idx >= len(self.symbols):
                print("Warning: bad atom index {atom_index}", file=sys.stderr)
                continue

            near_atoms |= {
                AtomRef(a)
                for a in self.get_atoms_near_point(
                    (
                        self.geometry[atom_idx * 3],
                        self.geometry[atom_idx * 3 + 1],
                        self.geometry[atom_idx * 3 + 2],
                    ),
                    threshold,
                )
            }

        return [
            FragmentRef(i)
            for (i, f) in enumerate(self.fragments)
            if (i != frag_idx and not near_atoms.isdisjoint(f))
        ]

    def extend(self, other: Self) -> None:
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
        if other.connectivity is not None:
            if self.connectivity is None:
                self.connectivity = []
            for bond in other.connectivity:
                new_bond = Bond(
                    AtomRef(bond.atom1.value + offset),
                    AtomRef(bond.atom2.value + offset),
                    bond.order,
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
        if other.fragment_formal_charges is not None:
            if self.fragment_formal_charges is None:
                self.fragment_formal_charges = []
            self.fragment_formal_charges.extend(other.fragment_formal_charges)

        if other.fragment_partial_charges is not None:
            if self.fragment_partial_charges is None:
                self.fragment_partial_charges = []
            self.fragment_partial_charges.extend(other.fragment_partial_charges)

    def new_topology_from_residue_subset(
        self, residue_subset: list["Residue"]
    ) -> "Topology":
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
            new_topology.geometry.extend(self.geometry[i * 3 : (i + 1) * 3])

        # Copy optional data
        if self.labels:
            new_topology.labels = [self.labels[i] for i in atom_indices]

        if self.partial_charges:
            new_topology.partial_charges = [
                self.partial_charges[i] for i in atom_indices
            ]

        if self.formal_charges:
            new_topology.formal_charges = [self.formal_charges[i] for i in atom_indices]

        if self.velocities:
            new_topology.velocities = []
            for i in atom_indices:
                new_topology.velocities.extend(self.velocities[i * 3 : (i + 1) * 3])

        # Copy connectivity (only bonds between atoms in subset)
        if self.connectivity:
            new_topology.connectivity = []
            for bond in self.connectivity:
                if bond.atom1.value in old_to_new and bond.atom2.value in old_to_new:
                    new_bond = Bond(
                        AtomRef(old_to_new[bond.atom1.value]),
                        AtomRef(old_to_new[bond.atom2.value]),
                        bond.order,
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
            "GLY": "G",
            "ALA": "A",
            "VAL": "V",
            "LEU": "L",
            "ILE": "I",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "ASN": "N",
            "GLN": "Q",
            "CYS": "C",
            "CYD": "C",
            "CYX": "C",
            "MET": "M",
            "PHE": "F",
            "TYR": "Y",
            "TYD": "Y",
            "TRP": "W",
            "ASP": "D",
            "ASH": "D",
            "GLU": "E",
            "GLH": "E",
            "HIS": "H",
            "HIN": "H",
            "HID": "H",
            "HIE": "H",
            "HIP": "H",
            "LYS": "K",
            "LYD": "K",
            "LYN": "K",
            "ARG": "R",
            "HYP": "O",
        }
        return mapping.get(self.value, "X")


class Residue:
    """A residue containing a list of atoms."""

    def __init__(self, atoms: list[AtomRef] | list[int] | None = None):
        # Store as list of integers to match JSON serialization
        if atoms is None:
            self.atoms = []
        else:
            self.atoms = [
                atom.value if isinstance(atom, AtomRef) else atom for atom in atoms
            ]

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
    residues: list[Residue] = field(default_factory=list)

    # Sequence names (e.g., amino acid names)
    seqs: list[str] = field(default_factory=list)

    # Sequence numbers
    seq_ns: list[int] = field(default_factory=list)

    # Insertion codes
    insertion_codes: list[str] = field(default_factory=list)

    # WARN: Deprecated
    labeled: list[ResidueRef] | None = None

    # WARN: Deprecated
    labels: list[list[str]] | None = None

    @staticmethod
    def from_json(json_content: str | Path | dict) -> "Residues":
        if isinstance(json_content, str):
            residues_data = json.loads(json_content)
        elif isinstance(json_content, Path):
            with json_content.open() as f:
                residues_data = json.load(f)
        elif isinstance(json_content, dict):
            residues_data = json_content
        else:
            print(
                "WARNING: Tried to load Residues from JSON but "
                "it wasn't a str, Path, or dict!"
            )
            residues_data = json_content

        residues = Residues()
        residues.residues = [Residue(res) for res in residues_data["residues"]]
        residues.seqs = residues_data["seqs"]
        residues.seq_ns = residues_data["seq_ns"]
        residues.insertion_codes = residues_data["insertion_codes"]

        return residues

    def check(self) -> None:
        """Validate the residues structure."""
        if len(self.seqs) != len(self.residues):
            raise ValueError(
                f"Seqs length {len(self.seqs)} != residues length {len(self.residues)}"
            )

        if len(self.seq_ns) != len(self.residues):
            raise ValueError(
                f"Seq_ns length {len(self.seq_ns)} != residues length {len(self.residues)}"
            )

        if len(self.insertion_codes) != len(self.residues):
            raise ValueError(
                f"Insertion codes length {len(self.insertion_codes)} != residues length {len(self.residues)}"
            )

    def is_amino_acid(self, index: int) -> bool:
        """Check if residue at index is an amino acid."""
        if index >= len(self.seqs):
            return False
        return AminoAcidSeq.is_amino_acid(self.seqs[index])

    def amino_acid_indices(self) -> list[int]:
        """Get indices of amino acid residues."""
        return [i for i in range(len(self.seqs)) if self.is_amino_acid(i)]

    def non_amino_acid_indices(self) -> list[int]:
        """Get indices of non-amino acid residues."""
        return [i for i in range(len(self.seqs)) if not self.is_amino_acid(i)]

    def extend(self, other: Self) -> None:
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

    def new_residues_from_subset(self, residue_refs: list[ResidueRef]) -> "Residues":
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
    """A chain containing a list of residues."""

    def __init__(self, residues: list[ResidueRef] | list[int] | None = None):
        # Store as list of integers to match JSON serialization
        if residues is None:
            self.residues = []
        else:
            self.residues = [
                res.value if isinstance(res, ResidueRef) else res for res in residues
            ]

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
    chains: list[Chain] = field(default_factory=list)

    # Optional alpha helix residues
    alpha_helices: list[ResidueRef] | None = None

    # Optional beta sheet residues
    beta_sheets: list[ResidueRef] | None = None

    # WARN: Deprecated
    labeled: list[ChainRef] | None = None

    # WARN: Deprecated
    labels: list[list[str]] | None = None

    @staticmethod
    def from_json(json_content: str | Path | dict) -> "Chains":
        if isinstance(json_content, str):
            chains_data = json.loads(json_content)
        elif isinstance(json_content, Path):
            with json_content.open() as f:
                chains_data = json.load(f)
        elif isinstance(json_content, dict):
            chains_data = json_content
        else:
            print(
                "WARNING: Tried to load Chains from JSON but "
                "it wasn't a str, Path, or dict!"
            )
            chains_data = json_content

        chains = Chains()

        chains.chains = [Chain(chain) for chain in chains_data["chains"]]

        if chains_data.get("alpha_helices"):
            chains.alpha_helices = [ResidueRef(r) for r in chains_data["alpha_helices"]]

        if chains_data.get("beta_sheets"):
            chains.beta_sheets = [ResidueRef(r) for r in chains_data["beta_sheets"]]

        if chains_data.get("labeled"):
            chains.labeled = [ChainRef(c) for c in chains_data["labeled"]]

        if chains_data.get("labels"):
            chains.labels = chains_data["labels"]

        return chains

    def check(self) -> None:
        """Validate the chains structure."""
        # Basic validation - more complex checks could be added
        pass

    def extend(self, other: Self) -> None:
        """Extend this chains collection with another."""
        # Calculate residue offset
        residue_offset = sum(len(chain.residues) for chain in self.chains)

        # Extend chains with renumbered residue references
        for chain in other.chains:
            new_residue_refs = [ref + residue_offset for ref in chain.residues]
            self.chains.append(Chain(new_residue_refs))

        # Extend secondary structure info
        if self.alpha_helices is not None and other.alpha_helices is not None:
            new_alpha_helices = [
                ref.value + residue_offset for ref in other.alpha_helices
            ]
            self.alpha_helices.extend([ResidueRef(ref) for ref in new_alpha_helices])

        if self.beta_sheets is not None and other.beta_sheets is not None:
            new_beta_sheets = [ref.value + residue_offset for ref in other.beta_sheets]
            self.beta_sheets.extend([ResidueRef(ref) for ref in new_beta_sheets])

    def new_chains_from_residue_subset(
        self, residue_refs: list[ResidueRef]
    ) -> "Chains":
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
            new_chain_residues.sort(
                key=lambda new_idx: original_chain.residues.index(
                    residue_refs[new_idx].value
                )
            )
            new_chains.chains.append(Chain(new_chain_residues))

        # Filter secondary structure info
        if self.alpha_helices:
            new_alpha_helices = []
            for residue_ref in self.alpha_helices:
                if residue_ref.value in old_to_new_residue:
                    new_alpha_helices.append(
                        ResidueRef(old_to_new_residue[residue_ref.value])
                    )
            new_chains.alpha_helices = new_alpha_helices if new_alpha_helices else None

        if self.beta_sheets:
            new_beta_sheets = []
            for residue_ref in self.beta_sheets:
                if residue_ref.value in old_to_new_residue:
                    new_beta_sheets.append(
                        ResidueRef(old_to_new_residue[residue_ref.value])
                    )
            new_chains.beta_sheets = new_beta_sheets if new_beta_sheets else None

        return new_chains


@dataclass
class TRC:
    """
    Combined Topology, Residues, and Chains structure.
    This is the main structure for representing molecular systems on the Rush platform.
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
                    raise ValueError(
                        f"Chain references invalid residue index: {residue_idx}"
                    )
                if residue_idx in residue_set:
                    raise ValueError(
                        f"Residue {residue_idx} appears in multiple chains"
                    )
                residue_set.add(residue_idx)

        if len(residue_set) != len(self.residues.residues):
            raise ValueError("Not all residues are assigned to chains")

    def extend(self, other: Self) -> None:
        """Extend this TRC with another TRC."""
        self.topology.extend(other.topology)
        self.residues.extend(other.residues)
        self.chains.extend(other.chains)

    def new_trc_from_residue_subset(self, residue_refs: list[ResidueRef]) -> "TRC":
        """Create new TRC from a subset of residue references."""
        # Get residue subset
        residue_subset = [self.residues.residues[ref.value] for ref in residue_refs]

        return TRC(
            topology=self.topology.new_topology_from_residue_subset(residue_subset),
            residues=self.residues.new_residues_from_subset(residue_refs),
            chains=self.chains.new_chains_from_residue_subset(residue_refs),
        )


@dataclass(frozen=True)
class ResidueId:
    """Unique identifier for a residue."""

    chain_id: str
    sequence_number: int
    insertion_code: str
    residue_name: str

    def __str__(self) -> str:
        return f"{self.chain_id}_{self.sequence_number:>9}_{self.insertion_code}_{self.residue_name}"
