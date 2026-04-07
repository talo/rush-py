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

from typing import NewType

import libqdx

# Ref types — distinct for type checking, plain int at runtime
AtomRef = NewType("AtomRef", int)
ResidueRef = NewType("ResidueRef", int)
ChainRef = NewType("ChainRef", int)
FragmentRef = NewType("FragmentRef", int)

# Re-export native types
TRC = libqdx.TRC
Topology = libqdx.Topology
Residues = libqdx.Residues
Chains = libqdx.Chains
Element = libqdx.Element
Bond = libqdx.Bond
BondOrder = libqdx.BondOrder
Stereochemistry = libqdx.Stereochemistry
AminoAcidSeq = libqdx.AminoAcidSeq
AlphaHelices = libqdx.AlphaHelices
HelixClass = libqdx.HelixClass
BetaSheets = libqdx.BetaSheets
StrandSense = libqdx.StrandSense
AtomCheckStrictness = libqdx.AtomCheckStrictness
