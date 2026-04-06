"""
Molecular structure types for Rush.

Core types are provided by the native ``libqdx`` extension and re-exported
here for convenience.  This module used to contain pure-Python dataclass
implementations; those have been replaced by opaque Rust-backed objects
from ``libqdx`` for performance and correctness.

Primary types
-------------
TRC
    Combined **Topology + Residues + Chains** structure -- the main
    representation for molecular systems on the Rush platform.  Construct
    via ``TRC.from_dict(d)`` or by loading a file through
    ``rush.convert.load_structure``.

Topology
    Per-atom information: element symbols, XYZ geometry (flat list,
    3 * n_atoms), optional atom labels, formal/partial charges, bond
    connectivity, velocities, and fragment assignments.

Residues
    Residue groupings over atoms -- sequence names (e.g. amino-acid
    three-letter codes), sequence numbers, insertion codes, and the
    mapping of which atoms belong to which residue.

Chains
    Chain groupings over residues, plus optional secondary-structure
    annotations (alpha helices and beta sheets).

Element & bond types
--------------------
Element
    Chemical element enum (H, He, Li, ..., Kr).  Integer-valued,
    matching atomic number.

Bond
    A bond between two atoms (atom indices + bond order).

BondOrder
    Bond order enum: Single, Double, Triple, OneAndAHalf (partial /
    amide), Ring (aromatic).

Stereochemistry
    Atom stereochemistry descriptor (R/S chirality, E/Z geometry, etc.).

Secondary structure
-------------------
HelixClass
    PDB helix classification (right-handed alpha, 3-10, pi, etc.).

StrandSense
    Parallel vs. anti-parallel strand orientation in a beta sheet.

AlphaHelices
    Collection of alpha-helix annotations for a structure.

BetaSheets
    Collection of beta-sheet annotations for a structure.

Reference / index types
-----------------------
AtomRef
    ``NewType`` over ``int`` -- a zero-based atom index.

ResidueRef
    ``NewType`` over ``int`` -- a zero-based residue index.

ChainRef
    ``NewType`` over ``int`` -- a zero-based chain index.

FragmentRef
    ``NewType`` over ``int`` -- a zero-based fragment index.

These are plain ``int`` at runtime but provide static-analysis
distinctness so that, e.g., an ``AtomRef`` is not accidentally used
where a ``ResidueRef`` is expected.

Quick examples
--------------
Loading a structure and inspecting it::

    from rush.convert import load_structure

    trc = load_structure("1crn.pdb")
    print(len(trc.topology.symbols))  # number of atoms
    print(trc.residues.seqs[:5])      # first five residue names
    print(len(trc.chains.chains))     # number of chains

Converting to/from JSON dicts::

    d = trc.to_dict()          # -> dict with topology/residues/chains
    trc2 = TRC.from_dict(d)    # round-trip back to TRC

Validation::

    trc.check()                # raises on inconsistent data
"""

from enum import Enum
from typing import NewType

import libqdx

# Ref types — distinct for type checking, plain int at runtime
AtomRef = NewType("AtomRef", int)
ResidueRef = NewType("ResidueRef", int)
ChainRef = NewType("ChainRef", int)
FragmentRef = NewType("FragmentRef", int)

# Re-export native types
TRC = libqdx.PyTRC
Topology = libqdx.PyTopology
Residues = libqdx.PyResidues
Chains = libqdx.PyChains
Element = libqdx.Element
Bond = libqdx.Bond
BondOrder = libqdx.BondOrder
Stereochemistry = libqdx.Stereochemistry
HelixClass = libqdx.HelixClass
StrandSense = libqdx.StrandSense
AlphaHelices = libqdx.AlphaHelices
BetaSheets = libqdx.BetaSheets
AtomCheckStrictness = libqdx.AtomCheckStrictness


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

    _SINGLE_LETTER = {
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

    def to_single_letter(self) -> str:
        """Convert to single letter code."""
        return self._SINGLE_LETTER.get(self.value, "X")
