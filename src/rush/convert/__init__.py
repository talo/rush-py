"""
Conversion utilities for molecular structure file formats.

This module provides functions to convert between PDB, mmCIF, and JSON formats.
"""

from pathlib import Path

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
    with path.open("r") as f:
        content = f.read()

    # Determine file type by extension
    suffix = path.suffix.lower()
    if suffix == ".json":
        return from_json(content)
    elif suffix in {".cif", ".mmcif"}:
        return from_mmcif(content)
    elif suffix == ".pdb":
        return from_pdb(content)
    else:
        # Try to guess from content
        content_lower = content.lower()
        if content.strip().startswith("[") or content.strip().startswith("{"):
            return from_json(content)
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
        content = to_json(trcs)
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


__all__ = [
    "from_json",
    "to_json",
    "from_mmcif",
    "from_pdb",
    "to_pdb",
    "from_sdf",
    "load_structure",
    "save_structure",
]
