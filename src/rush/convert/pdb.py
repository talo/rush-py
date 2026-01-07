"""
PDB file parsing and writing functionality.
"""

import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from ..mol import (
    TRC,
    AminoAcidSeq,
    AtomRef,
    Bond,
    BondOrder,
    Chain,
    ChainRef,
    Element,
    FormalCharge,
    Fragment,
    Residue,
    ResidueId,
    ResidueRef,
)


@dataclass
class PDBAtom:
    """Represents a parsed PDB ATOM/HETATM record."""

    atom_idx: int
    atom_name: str
    alternate_location: str | None
    residue_name: str
    chain_id: str
    sequence_number: int
    residue_insertion: str | None
    atom_x: float
    atom_y: float
    atom_z: float
    occupancy: float
    temperature_factor: float
    segment_id: str | None
    element_symbol: Element
    charge: int | None


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
        residue_insertion = (
            line[26].strip() if len(line) > 26 and line[26].strip() else None
        )

        atom_x = float(line[30:38].strip()) if line[30:38].strip() else 0.0
        atom_y = float(line[38:46].strip()) if line[38:46].strip() else 0.0
        atom_z = float(line[46:54].strip()) if line[46:54].strip() else 0.0

        occupancy = (
            float(line[54:60].strip())
            if len(line) > 60 and line[54:60].strip()
            else 1.0
        )
        temperature_factor = (
            float(line[60:66].strip())
            if len(line) > 66 and line[60:66].strip()
            else 0.0
        )

        segment_id = (
            line[72:76].strip() if len(line) > 76 and line[72:76].strip() else None
        )

        element_symbol_str = (
            line[76:78].strip()
            if len(line) > 78 and line[76:78].strip()
            else atom_name[0]
        )
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
            charge=charge,
        )
    except (ValueError, IndexError) as e:
        raise ValueError(f"Line {line_num}: Error parsing ATOM/HETATM line: {e}")


def _parse_conect_line(line: str) -> list[int]:
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


def _build_trc(
    atoms: list[PDBAtom],
    atom_ids: list[int],
    residue_data: OrderedDict,
    chain_data: dict[str, set[ResidueId]],
    connectivity: list[tuple[int, int, int]],
) -> TRC:
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
    trc.topology.formal_charges = [
        FormalCharge(charge) for charge in atom_formal_charges
    ]

    # Sort residues by ResidueId (chain_id, sequence_number, insertion_code, residue_name)
    # This matches the Rust BTreeMap ordering
    sorted_residue_ids = sorted(
        residue_data.keys(),
        key=lambda rid: (
            rid.chain_id,
            rid.sequence_number,
            rid.insertion_code,
            rid.residue_name,
        ),
    )

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
        insertion_code = (
            "" if residue_id.insertion_code == "~" else residue_id.insertion_code
        )
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
        sorted_residue_ids = sorted(
            chain_residue_ids, key=lambda rid: (rid.sequence_number, rid.insertion_code)
        )

        chain_residue_refs = [
            ResidueRef(residue_id_to_index[rid]) for rid in sorted_residue_ids
        ]
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
        bonds.append(
            Bond(
                AtomRef(min(origin_idx, target_idx)),
                AtomRef(max(origin_idx, target_idx)),
                BondOrder(order),
            )
        )
    trc.topology.connectivity = bonds

    # Calculate fragment formal charges (sum of atom charges in each residue)
    fragment_formal_charges = []
    for residue in trc.residues.residues:
        total_charge = sum(atom_formal_charges[atom_idx] for atom_idx in residue.atoms)
        fragment_formal_charges.append(FormalCharge(total_charge))
    trc.topology.fragment_formal_charges = fragment_formal_charges

    return trc


def _apply_global_connectivity(
    trc: TRC, atom_ids: list[int], global_connectivity: list[tuple[int, int, int]]
):
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
        additional_bonds.append(
            Bond(
                AtomRef(min(origin_idx, target_idx)),
                AtomRef(max(origin_idx, target_idx)),
                BondOrder(order),
            )
        )

    # Add to existing connectivity
    if trc.topology.connectivity:
        trc.topology.connectivity.extend(additional_bonds)
    else:
        trc.topology.connectivity = additional_bonds


def from_pdb(pdb_content: str) -> TRC | list[TRC]:
    """
    Parse PDB file content into TRC structures.

    Args:
        pdb_content: String content of a PDB file

    Returns:
        TRC structure or list of TRC structures (one per model in multi-model files)
    """
    trcs = []
    trc_atom_ids = []
    global_connectivity = []  # List of (origin, target, order) tuples

    lines = pdb_content.strip().split("\n")
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
                    if (
                        atom.alternate_location is None
                        or atom.alternate_location == "A"
                    ):
                        atoms.append(atom)
                        atom_ids.append(atom.atom_idx)

                        # Create residue identifier
                        # Note: insertion_code uses "~" for sorting (to sort after all letters)
                        # but the actual value stored in the residues structure is empty string
                        residue_id = ResidueId(
                            chain_id=atom.chain_id,
                            sequence_number=atom.sequence_number,
                            insertion_code=atom.residue_insertion or "~",
                            residue_name=atom.residue_name,
                        )

                        # Add to residue data
                        if residue_id not in residue_data:
                            residue_data[residue_id] = []
                        residue_data[residue_id].append(
                            len(atoms) - 1
                        )  # Index in atoms list

                        # Add to chain data
                        chain_data[atom.chain_id].add(residue_id)
                    # else: skip atoms with other alternate locations

                except ValueError as e:
                    print(f"Warning: {e}", file=sys.stderr)
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

    if len(trcs) == 1:
        return trcs[0]
    return trcs


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
        chain_id = chr(65 + chain_idx) if chain_idx < 26 else "A"  # A, B, C, ...

        residue_name = (
            trc.residues.seqs[residue_idx]
            if residue_idx < len(trc.residues.seqs)
            else "UNK"
        )
        seq_num = (
            trc.residues.seq_ns[residue_idx]
            if residue_idx < len(trc.residues.seq_ns)
            else 1
        )
        insertion_code = (
            trc.residues.insertion_codes[residue_idx]
            if residue_idx < len(trc.residues.insertion_codes)
            else ""
        )

        for atom_idx in residue.atoms:
            if atom_idx >= len(trc.topology.symbols):
                continue

            element = trc.topology.symbols[atom_idx]
            atom_name = (
                trc.topology.labels[atom_idx] if trc.topology.labels else str(element)
            )

            x = (
                trc.topology.geometry[atom_idx * 3]
                if atom_idx * 3 < len(trc.topology.geometry)
                else 0.0
            )
            y = (
                trc.topology.geometry[atom_idx * 3 + 1]
                if atom_idx * 3 + 1 < len(trc.topology.geometry)
                else 0.0
            )
            z = (
                trc.topology.geometry[atom_idx * 3 + 2]
                if atom_idx * 3 + 2 < len(trc.topology.geometry)
                else 0.0
            )

            formal_charge = 0
            if trc.topology.formal_charges and atom_idx < len(
                trc.topology.formal_charges
            ):
                formal_charge = trc.topology.formal_charges[atom_idx].charge

            # Format ATOM record
            record_type = (
                "ATOM" if AminoAcidSeq.is_amino_acid(residue_name) else "HETATM"
            )

            line = f"{record_type:<6}{atom_idx:>5} {atom_name:<4} {residue_name:>3} {chain_id}{seq_num:>4}{insertion_code:<1}   {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {str(element):>2}{formal_charge:+2d}"
            lines.append(line)
            atom_idx += 1

    lines.append("END")
    return "\n".join(lines)
