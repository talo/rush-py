"""
mmCIF file parsing functionality.
"""

from collections import OrderedDict, defaultdict

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
    ResidueId,
    ResidueRef,
)


def _parse_mmcif_value(value: str) -> str:
    """Parse an mmCIF value, handling quoted strings and special characters."""
    value = value.strip()
    if value in (".", "?"):
        return ""
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def _parse_mmcif_loop(
    lines: list[str], start_idx: int, prefix: str
) -> tuple[tuple[list[str], list[list[str]]] | None, int]:
    """
    Parse an mmCIF loop starting at start_idx.

    Returns:
        ((column_names, rows), next_idx) or (None, next_idx) if not a loop with the given prefix
    """
    i = start_idx

    # Check if this is a loop
    if i >= len(lines) or not lines[i].strip().startswith("loop_"):
        return (None, i)

    i += 1

    # Parse column names
    columns = []
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if not line.startswith("_"):
            break
        if line.startswith(prefix):
            columns.append(line[len(prefix) :])
        elif (
            columns
        ):  # Started collecting columns for this prefix, now hit a different prefix
            break
        i += 1

    if not columns:
        return (None, i)

    # Parse data rows (may span multiple lines)
    rows = []
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.startswith("_") or line.startswith("loop_"):
            break

        # Parse fields from current line (and additional lines if needed)
        fields = []
        current_line = lines[i]
        i += 1

        while len(fields) < len(columns):
            # Parse tokens from current_line
            tokens = []
            j = 0
            current_line_stripped = current_line.rstrip("\n\r")
            while j < len(current_line_stripped):
                # Skip whitespace
                while (
                    j < len(current_line_stripped) and current_line_stripped[j] in " \t"
                ):
                    j += 1
                if j >= len(current_line_stripped):
                    break

                # Check for quoted string
                if current_line_stripped[j] in ("'", '"'):
                    quote_char = current_line_stripped[j]
                    j += 1
                    start = j
                    while (
                        j < len(current_line_stripped)
                        and current_line_stripped[j] != quote_char
                    ):
                        j += 1
                    tokens.append(current_line_stripped[start:j])
                    j += 1  # Skip closing quote
                else:
                    # Unquoted value
                    start = j
                    while (
                        j < len(current_line_stripped)
                        and current_line_stripped[j] not in " \t"
                    ):
                        j += 1
                    tokens.append(current_line_stripped[start:j])

            fields.extend(tokens)

            # If we don't have enough fields yet, try to read the next line
            if len(fields) < len(columns):
                if i < len(lines):
                    next_line = lines[i].strip()
                    if (
                        next_line
                        and not next_line.startswith("_")
                        and not next_line.startswith("loop_")
                        and not next_line.startswith("data_")
                    ):
                        current_line = lines[i]
                        i += 1
                    else:
                        break
                else:
                    break

        if len(fields) == len(columns):
            rows.append(fields)

    return ((columns, rows), i)


def _build_trc_from_mmcif_atoms(
    atoms: list[dict],
    struct_conn_data: tuple[list[str], list[list[str]]] | None,
    comp_bond_data: tuple[list[str], list[list[str]]] | None,
) -> TRC:
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
        alt_id = atom["label_alt_id"]
        if alt_id and alt_id != "A":
            continue

        # Parse element from type_symbol
        type_symbol = atom["type_symbol"]
        # Remove non-alphabetic characters
        element_str = "".join(c for c in type_symbol if c.isalpha())
        try:
            element = Element.from_str(element_str)
        except (ValueError, KeyError):
            element = Element.C  # Default to carbon

        topology_idx = len(atom_symbols)
        atom_index_map[orig_idx] = topology_idx

        atom_symbols.append(element)
        geometry.extend([atom["Cartn_x"], atom["Cartn_y"], atom["Cartn_z"]])

        atom_ids.append(atom["id"])
        atom_labels.append(atom["label_atom_id"])
        atom_formal_charges.append(atom["pdbx_formal_charge"])

        # Create residue identifier using auth fields and "~" for sorting
        residue_id = ResidueId(
            chain_id=atom["auth_asym_id"],
            sequence_number=atom["auth_seq_id"],
            insertion_code=atom["pdbx_PDB_ins_code"] or "~",
            residue_name=atom["label_comp_id"],
        )

        if residue_id not in residue_data:
            residue_data[residue_id] = []
        residue_data[residue_id].append(len(atom_symbols) - 1)

        chain_data[atom["auth_asym_id"]].add(residue_id)

    # Build topology
    trc.topology.symbols = atom_symbols
    trc.topology.geometry = geometry
    trc.topology.labels = atom_labels
    trc.topology.formal_charges = [
        FormalCharge(charge) for charge in atom_formal_charges
    ]

    # Sort residues by ResidueId
    sorted_residue_ids = sorted(
        residue_data.keys(),
        key=lambda rid: (
            rid.chain_id,
            rid.sequence_number,
            rid.insertion_code,
            rid.residue_name,
        ),
    )

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
        sorted_chain_residue_ids = sorted(
            chain_residue_ids,
            key=lambda rid: (rid.sequence_number, rid.insertion_code, rid.residue_name),
        )

        chain_residue_refs = [
            ResidueRef(residue_id_to_index[rid])
            for rid in sorted_chain_residue_ids
            if rid in residue_id_to_index
        ]
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
                return ""

            def get_int_val(name: str) -> int:
                val = get_val(name)
                try:
                    return int(val) if val else 0
                except ValueError:
                    return 0

            # Find atoms by label (uses label_ fields, not auth_)
            ptnr1_atom = get_val("ptnr1_label_atom_id")
            ptnr1_asym = get_val("ptnr1_label_asym_id")
            ptnr1_seq = get_int_val("ptnr1_label_seq_id")
            ptnr2_atom = get_val("ptnr2_label_atom_id")
            ptnr2_asym = get_val("ptnr2_label_asym_id")
            ptnr2_seq = get_int_val("ptnr2_label_seq_id")
            conn_type = get_val("conn_type_id")

            # Find matching atoms using label_ fields (find FIRST match like Rust .position())
            atom1_orig_idx = None
            atom2_orig_idx = None
            for idx, atom in enumerate(atoms):
                if atom1_orig_idx is None and (
                    atom["label_atom_id"] == ptnr1_atom
                    and atom["label_asym_id"] == ptnr1_asym
                    and atom["label_seq_id"] == ptnr1_seq
                ):
                    atom1_orig_idx = idx
                if atom2_orig_idx is None and (
                    atom["label_atom_id"] == ptnr2_atom
                    and atom["label_asym_id"] == ptnr2_asym
                    and atom["label_seq_id"] == ptnr2_seq
                ):
                    atom2_orig_idx = idx
                if atom1_orig_idx is not None and atom2_orig_idx is not None:
                    break

            if atom1_orig_idx is not None and atom2_orig_idx is not None:
                topo_idx1 = atom_index_map.get(atom1_orig_idx)
                topo_idx2 = atom_index_map.get(atom2_orig_idx)

                if topo_idx1 is not None and topo_idx2 is not None:
                    bond_order = 1  # Default to single bond
                    if conn_type in ["covale", "metalc", "disulf"]:
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
                return ""

            comp_id = get_val("comp_id")
            atom_id_1 = get_val("atom_id_1")
            atom_id_2 = get_val("atom_id_2")
            value_order = get_val("value_order")

            comp_bonds[comp_id].append((atom_id_1, atom_id_2, value_order))

        # Group atoms by residue for efficient lookup
        # Note: Rust uses (comp_id, auth_asym_id, auth_seq_id) without insertion code
        residue_atoms = defaultdict(
            list
        )  # (comp_id, auth_asym_id, auth_seq_id) -> list of (orig_idx, topo_idx, atom)
        for orig_idx, atom in enumerate(atoms):
            if atom["label_alt_id"] == "" or atom["label_alt_id"] == "A":
                topo_idx = atom_index_map.get(orig_idx)
                if topo_idx is not None:
                    key = (
                        atom["label_comp_id"],
                        atom["auth_asym_id"],
                        atom["auth_seq_id"],
                    )
                    residue_atoms[key].append((orig_idx, topo_idx, atom))

        # Apply bond definitions to residues
        for (comp_id, chain_id, seq_id), res_atoms in residue_atoms.items():
            if comp_id in comp_bonds:
                for atom_id_1, atom_id_2, value_order in comp_bonds[comp_id]:
                    # Find THE FIRST atom that matches each atom_id (Rust uses find())
                    topo_idx1 = None
                    topo_idx2 = None
                    for _, topo_idx, atom in res_atoms:
                        if topo_idx1 is None and atom["label_atom_id"] == atom_id_1:
                            topo_idx1 = topo_idx
                        if topo_idx2 is None and atom["label_atom_id"] == atom_id_2:
                            topo_idx2 = topo_idx
                        if topo_idx1 is not None and topo_idx2 is not None:
                            break

                    if topo_idx1 is not None and topo_idx2 is not None:
                        # Parse bond order
                        bond_order = 1
                        if value_order == "SING":
                            bond_order = 1
                        elif value_order == "DOUB":
                            bond_order = 2
                        elif value_order == "TRIP":
                            bond_order = 3
                        elif value_order == "QUAD":
                            bond_order = 4
                        elif value_order == "AROM":
                            bond_order = 5

                        min_idx = min(topo_idx1, topo_idx2)
                        max_idx = max(topo_idx1, topo_idx2)
                        connectivity_deduper[(min_idx, max_idx)] = bond_order

    # Convert to Bond objects
    bonds = []
    for (min_idx, max_idx), order in sorted(connectivity_deduper.items()):
        bonds.append(Bond(AtomRef(min_idx), AtomRef(max_idx), BondOrder(order)))
    trc.topology.connectivity = bonds

    return trc


def from_mmcif(mmcif_content: str) -> TRC | list[TRC]:
    """
    Parse mmCIF file contents into TRC structures.

    Args:
        mmcif_content: String contents of an mmCIF file

    Returns:
        TRC structure or list of TRC structures
    """
    lines = mmcif_content.split("\n")
    trcs = []

    # Parse loops
    models = defaultdict(list)  # model_num -> list of atoms
    atom_loop_data = None
    struct_conn_data = None
    comp_bond_data = None

    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("loop_"):
            # Try to parse atom_site loop
            result, next_i = _parse_mmcif_loop(lines, i, "_atom_site.")
            if result:
                columns, rows = result
                # Check if this has atom_site columns
                if any("id" in col or "type_symbol" in col for col in columns):
                    atom_loop_data = (columns, rows)
                i = next_i
                continue

            # Try to parse struct_conn loop
            result, next_i = _parse_mmcif_loop(lines, i, "_struct_conn.")
            if result:
                struct_conn_data = result
                i = next_i
                continue

            # Try to parse chem_comp_bond loop
            result, next_i = _parse_mmcif_loop(lines, i, "_chem_comp_bond.")
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

        def get_val(name: str, default: str = "") -> str:
            idx = col_idx.get(name)
            if idx is not None and idx < len(row):
                val = _parse_mmcif_value(row[idx])
                return val if val else default
            return default

        def get_int(name: str, default: int = 0) -> int | None:
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
        auth_seq_id_val = get_int("auth_seq_id")
        if auth_seq_id_val is None:
            auth_seq_id_val = get_int("label_seq_id")
            if auth_seq_id_val is None:
                auth_seq_id_val = 0

        atom = {
            "id": get_int_with_default("id", 0),
            "type_symbol": get_val("type_symbol", "C"),
            "label_atom_id": get_val("label_atom_id", "C"),
            "label_alt_id": get_val("label_alt_id", ""),
            "label_comp_id": get_val("label_comp_id", "UNK"),
            "label_asym_id": get_val("label_asym_id", "A"),
            "label_seq_id": get_int_with_default("label_seq_id", 0),
            "pdbx_PDB_ins_code": get_val("pdbx_PDB_ins_code", ""),
            "Cartn_x": get_float("Cartn_x", 0.0),
            "Cartn_y": get_float("Cartn_y", 0.0),
            "Cartn_z": get_float("Cartn_z", 0.0),
            "occupancy": get_float("occupancy", 1.0),
            "B_iso_or_equiv": get_float("B_iso_or_equiv", 0.0),
            "pdbx_formal_charge": get_int_with_default("pdbx_formal_charge", 0),
            "auth_asym_id": (
                get_val("auth_asym_id", "") or get_val("label_asym_id", "A")
            ),
            "auth_seq_id": auth_seq_id_val,
            "group_PDB": get_val("group_PDB", "ATOM"),
            "pdbx_PDB_model_num": get_val("pdbx_PDB_model_num", "1"),
        }

        model_num = atom["pdbx_PDB_model_num"]
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

    if len(trcs) == 1:
        return trcs[0]
    return trcs
