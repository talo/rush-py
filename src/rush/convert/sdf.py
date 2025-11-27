"""
SDF file parsing functionality.

Converts SDF (Structure Data File) format to TRC structures.
Supports SDF V2000 format.
"""

from enum import Enum
from typing import Any

from ..mol import (
    TRC,
    AtomRef,
    Bond,
    BondOrder,
    Chain,
    ChainRef,
    Element,
    FormalCharge,
    Fragment,
    Residue,
    ResidueRef,
)


class SDFParseState(Enum):
    """Parser state machine states."""

    HEADER_BLOCK = "HeaderBlock"
    COUNTS_LINE = "CountsLine"
    ATOM_BLOCK = "AtomBlock"
    BOND_BLOCK = "BondBlock"
    PROPERTIES_BLOCK = "PropertiesBlock"
    DATA_ITEMS = "DataItems"
    DONE = "Done"


class SDFPropertyType(Enum):
    """SDF property types."""

    CHARGE = "CHG"
    END = "END"
    UNK = "Unk"


# SDF bond types: 1=single, 2=double, 3=triple, 4=aromatic/ring
_SDF_BOND_TYPES = [1, 2, 3, 4]


def _charge_field_to_charge(c: int) -> int | None:
    """Convert SDF charge field to actual charge value."""
    charge_map = {
        0: 0,
        1: 3,
        2: 2,
        3: 1,
        5: -1,
        6: -2,
        7: -3,
    }
    return charge_map.get(c)


def _bond_order_from_sdf(order: int) -> BondOrder:
    """Convert SDF bond type to BondOrder."""
    if order == 1:
        return BondOrder.Single
    elif order == 2:
        return BondOrder.Double
    elif order == 3:
        return BondOrder.Triple
    elif order == 4:
        return BondOrder.Ring
    else:
        raise ValueError(f"Invalid bond type: {order}")


def _parse_sdf_entry(sdf_content: str) -> dict[str, Any]:
    """
    Parse a single SDF entry into a molecule dictionary.

    SDF V2000 format:
    - Line 1: Molecule name
    - Line 2: User/Program name
    - Line 3: Comment
    - Line 4: Counts line (num_atoms num_bonds ...)
    - Lines 5-4+num_atoms: Atom block (x y z symbol ...)
    - Lines 5+num_atoms-4+num_atoms+num_bonds: Bond block
    - Properties block (optional, e.g., CHG for charges)
    - Data items (optional, e.g., SMILES)
    - Terminator: "$$$$"
    """
    state = SDFParseState.HEADER_BLOCK
    seen_chg_property = False
    num_atoms = 0
    num_bonds = 0

    molecule = {
        "name": "",
        "atoms": [],
        "bonds": [],
        "associated_data": [],
    }

    lines = sdf_content.split("\n")
    line_number = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        line_number = i + 1

        # Skip empty lines (except in header block)
        if not line.strip() and state != SDFParseState.HEADER_BLOCK:
            i += 1
            continue

        if state == SDFParseState.HEADER_BLOCK:
            molecule["name"] = line.strip()
            # Skip next two lines (user/program and comment)
            if i + 2 >= len(lines):
                raise ValueError(f"Line {line_number + 1}: Missing header lines")
            i += 3  # Skip header + 2 comment lines
            state = SDFParseState.COUNTS_LINE
            continue

        elif state == SDFParseState.COUNTS_LINE:
            if "V3000" in line:
                raise ValueError(f"Line {line_number}: V3000 format not supported")

            if len(line) < 6:
                raise ValueError(f"Line {line_number}: Counts line too short")

            try:
                num_atoms = int(line[:3].strip())
                num_bonds = int(line[3:6].strip())
            except ValueError as e:
                raise ValueError(f"Line {line_number}: Could not parse counts: {e}")

            molecule["atoms"] = []
            molecule["bonds"] = []

            state = SDFParseState.ATOM_BLOCK
            i += 1
            continue

        elif state == SDFParseState.ATOM_BLOCK:
            if len(line) < 39:
                raise ValueError(f"Line {line_number}: Atom line too short")

            try:
                x = float(line[0:10].strip())
                y = float(line[10:20].strip())
                z = float(line[20:30].strip())
                symbol = line[30:33].strip()

                # Mass difference (optional, at position 33-35)
                # TODO: never used
                _mass_diff = 0
                if len(line) >= 35:
                    try:
                        _mass_diff = int(line[33:35].strip() or "0")
                    except ValueError:
                        pass

                # Charge (at position 36-39, but SDF uses special encoding)
                charge = 0
                if len(line) >= 39:
                    try:
                        charge_field = int(line[36:39].strip() or "0")
                        charge = _charge_field_to_charge(charge_field)
                        if charge is None:
                            charge = 0
                    except ValueError:
                        charge = 0

                molecule["atoms"].append(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "symbol": symbol,
                        "charge": charge,
                    }
                )

                if len(molecule["atoms"]) >= num_atoms:
                    if num_bonds == 0:
                        state = SDFParseState.PROPERTIES_BLOCK
                    else:
                        state = SDFParseState.BOND_BLOCK

            except (ValueError, IndexError) as e:
                raise ValueError(f"Line {line_number}: Could not parse atom: {e}")

            i += 1
            continue

        elif state == SDFParseState.BOND_BLOCK:
            if len(line) < 9:
                raise ValueError(f"Line {line_number}: Bond line too short")

            try:
                atom1 = (
                    int(line[0:3].strip()) - 1
                )  # SDF is 1-indexed, convert to 0-indexed
                atom2 = int(line[3:6].strip()) - 1
                bond_type = int(line[6:9].strip())
                bond_stereo = 0
                if len(line) >= 12:
                    try:
                        bond_stereo = int(line[9:12].strip() or "0")
                    except ValueError:
                        pass

                if bond_type not in _SDF_BOND_TYPES:
                    raise ValueError(
                        f"Line {line_number}: Invalid bond type: {bond_type}"
                    )

                molecule["bonds"].append(
                    {
                        "atom1": atom1,
                        "atom2": atom2,
                        "bond_type": bond_type,
                        "bond_stereo": bond_stereo,
                    }
                )

                if len(molecule["bonds"]) >= num_bonds:
                    state = SDFParseState.PROPERTIES_BLOCK

            except (ValueError, IndexError) as e:
                raise ValueError(f"Line {line_number}: Could not parse bond: {e}")

            i += 1
            continue

        elif state == SDFParseState.PROPERTIES_BLOCK:
            if len(line) < 6:
                # Might be empty line or start of data items
                if not line.strip():
                    state = SDFParseState.DATA_ITEMS
                    i += 1
                    continue
                else:
                    state = SDFParseState.DATA_ITEMS
                    continue

            try:
                prop_type_str = line[3:6].strip()
                if prop_type_str == "CHG":
                    prop_type = SDFPropertyType.CHARGE
                elif prop_type_str == "END":
                    prop_type = SDFPropertyType.END
                else:
                    prop_type = SDFPropertyType.UNK

                if prop_type == SDFPropertyType.CHARGE:
                    if not seen_chg_property:
                        # Reset all charges to 0
                        for atom in molecule["atoms"]:
                            atom["charge"] = 0
                        seen_chg_property = True

                    # Parse charge count (position 6-8 or 6-9)
                    # apparently, the count block is 6..9 in some standards, and 6..8 in others
                    # lets determine this by checking if 9 is a space or not
                    count_end = 9 if len(line) > 8 and line[8:9].strip() else 8
                    if len(line) < count_end:
                        raise ValueError(f"Line {line_number}: CHG line too short")

                    count = int(line[6:count_end].strip())

                    if count == 0 or count > 8:
                        raise ValueError(
                            f"Line {line_number}: CHG count out of range: {count}"
                        )

                    # Parse charge entries from the same line
                    # Each entry is 8 characters: 4 for index, 4 for charge
                    for j in range(count):
                        start = count_end + 8 * j
                        end = count_end + 4 + 8 * j

                        if len(line) < end:
                            raise ValueError(
                                f"Line {line_number}: CHG entry out of range"
                            )

                        atom_idx = (
                            int(line[start:end].strip()) - 1
                        )  # 1-indexed to 0-indexed

                        start = count_end + 4 + 8 * j
                        end = count_end + 8 + 8 * j

                        if len(line) < end:
                            raise ValueError(
                                f"Line {line_number}: CHG entry out of range"
                            )

                        charge = int(line[start:end].strip())

                        if atom_idx < 0 or atom_idx >= len(molecule["atoms"]):
                            raise ValueError(
                                f"Line {line_number}: CHG atom index out of range: {atom_idx}"
                            )
                        if charge < -3 or charge > 3:
                            raise ValueError(
                                f"Line {line_number}: CHG charge out of range: {charge}"
                            )

                        molecule["atoms"][atom_idx]["charge"] = charge

                elif prop_type == SDFPropertyType.END:
                    state = SDFParseState.DATA_ITEMS

            except (ValueError, IndexError):
                # If we can't parse as property, assume we're in data items
                state = SDFParseState.DATA_ITEMS
                continue

            i += 1
            continue

        elif state == SDFParseState.DATA_ITEMS:
            if line.strip() == "$$$$":
                # Terminator found
                break

            if line.startswith(">"):
                # Data item key
                start = line.find("<")
                if start == -1:
                    raise ValueError(f"Line {line_number}: Invalid data item format")
                end = line.find(">", start)
                if end == -1:
                    raise ValueError(f"Line {line_number}: Invalid data item format")

                key = line[start + 1 : end]
                data = []

                # Read data until empty line
                i += 1
                while i < len(lines):
                    data_line = lines[i]
                    if not data_line.strip():
                        break
                    data.append(data_line)
                    i += 1

                molecule["associated_data"].append((key, "\n".join(data)))
                continue

            i += 1
            continue

    return molecule


def _sdf_entries(sdf_content: str) -> list[tuple[int, str]]:
    """Split SDF content into individual entries (separated by $$$$)."""
    entries = []
    tail = sdf_content
    current_line_number = 1

    while True:
        terminator_pos = tail.find("\n$$$$")
        if terminator_pos == -1:
            # Check if there's a $$$$ at the end
            if tail.strip().endswith("$$$$"):
                entries.append((current_line_number, tail))
            else:
                raise ValueError(
                    f"Line {current_line_number}: Missing SDF terminator ($$$$)"
                )
            break

        offset_after_terminator = terminator_pos + 5
        if len(tail) == offset_after_terminator:
            entries.append((current_line_number, tail))
            break
        elif (
            len(tail) > offset_after_terminator
            and tail[offset_after_terminator] != "\n"
        ):
            raise ValueError(f"Line {current_line_number}: Invalid terminator format")
        else:
            entry = tail[: terminator_pos + 6]  # Include \n$$$$
            tail = tail[terminator_pos + 6 :]
            entries.append((current_line_number, entry))
            current_line_number += entry.count("\n")
            if not tail.strip():
                break

    return entries


def _molecule_to_trc(molecule: dict[str, Any]) -> TRC:
    """
    Convert a parsed molecule to TRC structure.

    Creates a TRC with:
    - Single residue containing all atoms
    - Residue name from molecule name (or "LIG" if empty)
    - Single chain containing that residue
    - Bonds as connectivity
    - Charges as formal_charges
    """
    trc = TRC()

    num_atoms = len(molecule["atoms"])

    # Build topology
    symbols = []
    geometry = []
    formal_charges = []
    labels = []

    # Track element counts for labeling (e.g., C1, C2, N1, H1, H2...)
    element_counts = {}

    for atom in molecule["atoms"]:
        try:
            element = Element.from_str(atom["symbol"])
        except ValueError:
            element = Element.C  # Default to carbon if unknown
        symbols.append(element)
        geometry.extend([float(atom["x"]), float(atom["y"]), float(atom["z"])])
        formal_charges.append(FormalCharge(atom["charge"]))

        # Create label based on element symbol and sequence number for that element
        element_symbol = atom["symbol"]
        element_counts[element_symbol] = element_counts.get(element_symbol, 0) + 1
        labels.append(f"{element_symbol}{element_counts[element_symbol]}")

    trc.topology.symbols = symbols
    trc.topology.geometry = geometry
    trc.topology.formal_charges = formal_charges
    trc.topology.labels = labels

    # Build connectivity (bonds) - ensure atom1 < atom2 for canonical ordering
    bonds = []
    for bond in molecule["bonds"]:
        atom1_idx = bond["atom1"]
        atom2_idx = bond["atom2"]
        bond_order = _bond_order_from_sdf(bond["bond_type"])

        # Ensure atom1 < atom2 (canonical ordering)
        if atom1_idx > atom2_idx:
            atom1_idx, atom2_idx = atom2_idx, atom1_idx

        bonds.append(
            Bond(
                AtomRef(atom1_idx),
                AtomRef(atom2_idx),
                bond_order,
            )
        )

    trc.topology.connectivity = bonds

    # Fragments: single fragment with all atoms
    trc.topology.fragments = [Fragment([AtomRef(i) for i in range(num_atoms)])]

    # Fragment formal charge: sum of all atom charges
    fragment_formal_charge = sum(atom["charge"] for atom in molecule["atoms"])
    trc.topology.fragment_formal_charges = [FormalCharge(fragment_formal_charge)]

    # Build residues: single residue with all atoms
    trc.residues.residues = [Residue([AtomRef(i) for i in range(num_atoms)])]
    trc.residues.seqs = ["LIG"]
    trc.residues.seq_ns = [0]
    trc.residues.insertion_codes = [""]
    # Label the ligand residue
    trc.residues.labeled = [ResidueRef(0)]
    trc.residues.labels = [[molecule["name"].strip().lower() or "LIG"]]

    # Build chains: single chain with residue 0
    trc.chains.chains = [Chain([ResidueRef(0)])]
    trc.chains.labeled = [ChainRef(0)]
    trc.chains.labels = [[molecule["name"].strip() or "LIG"]]

    return trc


def from_sdf(sdf_content: str) -> TRC | list[TRC]:
    """
    Parse SDF file contents into TRC structures.

    Args:
        sdf_content: SDF file content as string

    Returns:
        TRC structure or list of TRC structures (one per molecule in the SDF file)

    Raises:
        ValueError: If SDF parsing fails
    """
    entries = _sdf_entries(sdf_content)
    trcs: list[TRC] = []

    for line_number, entry in entries:
        try:
            molecule = _parse_sdf_entry(entry)
            trcs.append(_molecule_to_trc(molecule))
        except Exception as e:
            raise ValueError(
                f"Error parsing SDF entry starting at line {line_number}: {e}"
            )

    if len(trcs) == 1:
        return trcs[0]
    return trcs
