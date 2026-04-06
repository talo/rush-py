"""
Molecular structure types for Rush.

Core types are provided by the native ``libqdx`` extension.
Re-exported here for convenience.
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
