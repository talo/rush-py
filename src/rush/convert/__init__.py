"""
Conversion utilities for molecular structure file formats.

This module provides functions to convert between PDB, mmCIF, SDF, and QDX's TRC JSON formats.
"""

import copy
import json as std_json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeGuard

from ..mol import TRC
from .json import from_json, to_json
from .mmcif import from_mmcif
from .pdb import from_pdb, to_pdb
from .sdf import from_sdf


def load_structure(file_path: str | Path) -> TRC | list[TRC]:
    """
    Load structure from PDB, mmCIF, or JSON file.

    Args:
        file_path: Path to structure file

    Returns:
        TRC structure or list of TRC structures
    """
    path = Path(file_path)
    # Determine file type by extension
    suffix = path.suffix.lower()
    if suffix == ".json":
        return from_json(path)

    with path.open("r") as f:
        content = f.read()
    if suffix in {".cif", ".mmcif"}:
        return from_mmcif(content)
    elif suffix == ".pdb":
        return from_pdb(content)
    else:
        # Try to guess from content
        content_lower = content.lower()
        if content.strip().startswith("[") or content.strip().startswith("{"):
            return from_json(std_json.loads(content))
        elif "data_" in content_lower and "_atom_site" in content_lower:
            return from_mmcif(content)
        else:
            return from_pdb(content)


def save_structure(
    trcs: TRC | list[TRC], file_path: str | Path, format: str | None = None
):
    """
    Save TRC structures to file.

    Args:
        trcs: TRC structure or list of TRC structures
        file_path: Output file path
        format: Output format ('pdb', 'json', or None for auto-detect from extension)
    """
    path = Path(file_path)
    if format is None:
        # Auto-detect from extension
        if path.suffix.lower() == ".json":
            format = "json"
        elif path.suffix.lower() == ".pdb":
            format = "pdb"
        else:
            format = "pdb"  # Default

    if format.lower() == "json":
        with path.open("w") as f:
            std_json.dump(to_json(trcs), f, indent=2)
        return
    elif format.lower() == "pdb":
        if isinstance(trcs, TRC):
            trcs = [trcs]
        if len(trcs) > 1:
            # Multi-model PDB
            content_parts = []
            for i, trc in enumerate(trcs, 1):
                content_parts.append(f"MODEL     {i:>4}")
                content_parts.append(to_pdb(trc).replace("END\n", ""))
                content_parts.append("ENDMDL")
            content_parts.append("END")
            content = "\n".join(content_parts)
        else:
            content = to_pdb(trcs[0])
    else:
        raise ValueError(f"Unsupported format: {format}")

    with path.open("w") as f:
        f.write(content)


TrcInput = TRC | str | Path
TrcInputSeq = Sequence[TrcInput]


def _single_trc(trc: TRC | list[TRC], label: str | Path) -> TRC:
    if _is_trc_list(trc):
        if len(trc) != 1:
            raise ValueError(f"Expected 1 TRC in {label}, found {len(trc)}")
        return trc[0]
    if isinstance(trc, list):
        raise TypeError("Expected TRC list elements to be TRC objects")
    return trc


def _normalize_trc_inputs(inputs: tuple[TrcInput | TrcInputSeq, ...]) -> list[TrcInput]:
    normalized: list[TrcInput] = []
    for item in inputs:
        if isinstance(item, Sequence) and not isinstance(item, (str, Path, TRC)):
            normalized.extend(item)
        else:
            normalized.append(item)
    return normalized


def _is_trc_list(value: object) -> TypeGuard[list[TRC]]:
    return isinstance(value, list) and all(isinstance(item, TRC) for item in value)


def _load_trc(trc: TrcInput) -> TRC:
    """Load TRC from TRC object or file path."""
    if isinstance(trc, TRC):
        return trc
    if isinstance(trc, (str, Path)):
        path = Path(trc)
        if not path.exists():
            raise FileNotFoundError(f"TRC file not found: {trc}")
        loaded: TRC | list[TRC] = from_json(path)
        if _is_trc_list(loaded):
            if len(loaded) == 1:
                return loaded[0]
            merged = copy.deepcopy(loaded[0])
            for next_trc in loaded[1:]:
                merged.extend(next_trc)
            return merged
        if isinstance(loaded, list):
            raise TypeError("Expected TRC list elements to be TRC objects")
        return loaded
    raise TypeError(f"TRC must be a TRC object or file path, got {type(trc)}")


def merge_trcs(
    *trcs: TrcInput | TrcInputSeq,
    output_file: str | Path | None = None,
    skip_validation: bool = False,
) -> TRC:
    """
    Merge TRC objects into a single TRC.

    A TRC (Topology-Residues-Chains) object contains:
    - topology: atom information (symbols, geometry, bonds, charges, etc.)
    - residues: residue information (which atoms belong to which residues)
    - chains: chain information (which residues belong to which chains)

    When merging, atom indices, residue indices, and chain indices are renumbered
    to ensure uniqueness in the merged structure.

    Args:
        trcs: TRC objects or file paths. If a single list/tuple is provided,
            it is treated as the full set of inputs.
        output_file: Optional path to write the merged TRC JSON.
        skip_validation: If True, skip validation of the merged TRC.

    Returns:
        Merged TRC object.

    Raises:
        ValueError: If no inputs are provided or validation fails.
        FileNotFoundError: If file paths are provided but files don't exist.
    """
    trc_inputs = _normalize_trc_inputs(trcs)

    if not trc_inputs:
        raise ValueError("Expected at least one TRC input, found 0")

    merged: TRC | None = None
    for trc in trc_inputs:
        trc_obj = _load_trc(trc)
        if merged is None:
            merged = copy.deepcopy(trc_obj)
        else:
            merged.extend(trc_obj)

    if merged is None:
        raise ValueError("Expected at least one TRC input, found 0")

    if not skip_validation:
        merged.check()

    if output_file is not None:
        output_path = Path(output_file)
        with output_path.open("w") as f:
            std_json.dump(to_json([merged]), f, indent=2)

    return merged


__all__ = [
    "from_json",
    "to_json",
    "from_mmcif",
    "from_pdb",
    "to_pdb",
    "from_sdf",
    "load_structure",
    "save_structure",
    "merge_trcs",
]
