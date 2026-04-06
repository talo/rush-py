"""
Conversion utilities for molecular structure file formats.

Format parsing and writing are backed by the native libqdx Rust library.
"""

import json as std_json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeGuard

import libqdx

from ..mol import TRC
from .json import from_json, to_dict


def from_pdb(pdb_content: str) -> TRC | list[TRC]:
    """Parse PDB file content into TRC structures.

    Args:
        pdb_content: Raw PDB file text.

    Returns:
        A single TRC if the file contains one model, otherwise a list of TRCs.
    """
    trcs = libqdx.from_pdb(pdb_content)
    return trcs[0] if len(trcs) == 1 else trcs


def to_pdb(trc: TRC) -> str:
    """Convert a TRC structure to PDB format text.

    Args:
        trc: TRC structure to serialise.

    Returns:
        PDB-formatted string (includes trailing END record).
    """
    return libqdx.to_pdb(trc)


def from_mmcif(mmcif_content: str) -> TRC | list[TRC]:
    """Parse mmCIF file contents into TRC structures.

    Args:
        mmcif_content: Raw mmCIF file text.

    Returns:
        A single TRC if the file contains one model, otherwise a list of TRCs.
    """
    trcs = libqdx.from_mmcif(mmcif_content)
    return trcs[0] if len(trcs) == 1 else trcs


def from_sdf(sdf_content: str) -> TRC | list[TRC]:
    """Parse SDF file contents into TRC structures.

    Args:
        sdf_content: Raw SDF / MOL file text.

    Returns:
        A single TRC if the file contains one molecule, otherwise a list of TRCs.
    """
    trcs = libqdx.from_sdf(sdf_content)
    return trcs[0] if len(trcs) == 1 else trcs


def load_structure(file_path: str | Path) -> TRC | list[TRC]:
    """Load a molecular structure from a file.

    Supported formats: PDB, mmCIF (.cif / .mmcif), SDF, and TRC JSON.
    The format is determined by extension; when the extension is
    unrecognised the content is inspected heuristically.

    Args:
        file_path: Path to the structure file.

    Returns:
        A single TRC when the file contains one model/molecule, otherwise
        a list of TRCs.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return from_json(path)

    with path.open("r") as f:
        content = f.read()
    if suffix in {".cif", ".mmcif"}:
        return from_mmcif(content)
    elif suffix == ".pdb":
        return from_pdb(content)
    elif suffix == ".sdf":
        return from_sdf(content)
    else:
        # Unrecognised extension — try to guess from content
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
    """Save TRC structures to a file.

    Args:
        trcs: TRC structure or list of TRC structures to write.
        file_path: Output file path.
        format: Output format (``'pdb'`` or ``'json'``).  When *None* the
            format is inferred from the file extension.

    Raises:
        ValueError: If the format cannot be inferred or is unsupported.
    """
    path = Path(file_path)
    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix == ".pdb":
            format = "pdb"
        else:
            raise ValueError(
                f"Cannot infer format from extension '{suffix}'; pass format= explicitly"
            )

    if format.lower() == "json":
        with path.open("w") as f:
            std_json.dump(to_dict(trcs), f, indent=2)
        return
    elif format.lower() == "pdb":
        if isinstance(trcs, TRC):
            trcs = [trcs]
        if len(trcs) > 1:
            # Multi-model PDB: wrap each TRC in MODEL/ENDMDL records
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
        loaded = load_structure(trc)
        if _is_trc_list(loaded):
            if len(loaded) == 1:
                return loaded[0]
            merged = loaded[0]
            for next_trc in loaded[1:]:
                merged = merged.extend(next_trc)
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
    Merge one or more TRC objects (or file paths) into a single TRC.

    Atom, residue, and chain indices are renumbered so that the merged
    structure has unique indices throughout.

    Args:
        trcs: TRC objects or file paths.  A single list/tuple is treated
            as the full set of inputs.
        output_file: Optional path to write the merged TRC as JSON.
        skip_validation: If *True*, skip ``trc.check()`` on the result.

    Returns:
        The merged TRC object.

    Raises:
        ValueError: If no inputs are provided or validation fails.
        FileNotFoundError: If a file path does not exist.
    """
    trc_inputs = _normalize_trc_inputs(trcs)

    if not trc_inputs:
        raise ValueError("Expected at least one TRC input, found 0")

    merged = _load_trc(trc_inputs[0])
    for trc in trc_inputs[1:]:
        merged = merged.extend(_load_trc(trc))

    if not skip_validation:
        merged.check()

    if output_file is not None:
        output_path = Path(output_file)
        with output_path.open("w") as f:
            std_json.dump(to_dict([merged]), f, indent=2)

    return merged


__all__ = [
    "from_json",
    "to_dict",
    "from_mmcif",
    "from_pdb",
    "to_pdb",
    "from_sdf",
    "load_structure",
    "save_structure",
    "merge_trcs",
]
