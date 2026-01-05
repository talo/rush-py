"""
TRC merging functionality.
This module provides functions to merge TRC (Topology-Residues-Chains) structures,
which contain protein and/or ligand conformer information.
"""

import copy
from pathlib import Path
from typing import List, Optional, Union

from ..convert import from_json, to_json
from ..mol import TRC


def merge_trcs(
    trc1: Union[TRC, str, Path],
    trc2: Union[TRC, str, Path],
    skip_validation: bool = False,
) -> TRC:
    """
    Merge two TRC objects into a single TRC.
    A TRC (Topology-Residues-Chains) object contains:
    - topology: atom information (symbols, geometry, bonds, charges, etc.)
    - residues: residue information (which atoms belong to which residues)
    - chains: chain information (which residues belong to which chains)
    When merging, atom indices, residue indices, and chain indices are renumbered
    to ensure uniqueness in the merged structure.
    Args:
        trc1: First TRC object, or path to JSON file containing TRC or array of TRCs
        trc2: Second TRC object, or path to JSON file containing TRC or array of TRCs
        skip_validation: If True, skip validation of the merged TRC
    Returns:
        Merged TRC object
    Raises:
        ValueError: If validation fails and skip_validation is False
        FileNotFoundError: If file paths are provided but files don't exist
    """
    # Load TRCs if they are file paths
    trc1_obj = _load_trc(trc1)
    trc2_obj = _load_trc(trc2)

    # Create a deep copy of trc1 to avoid mutating the original
    merged = copy.deepcopy(trc1_obj)

    # Merge trc2 into merged using the extend method
    merged.extend(trc2_obj)

    # Validate if requested
    if not skip_validation:
        merged.check()

    return merged


def merge_trcs_from_files(
    input_files: List[Union[str, Path]],
    output_file: Optional[Union[str, Path]] = None,
    skip_validation: bool = False,
) -> TRC:
    """
    Merge multiple TRC files into a single TRC.
    This function reads multiple TRC JSON files (each can contain a single TRC
    or an array of TRCs), merges them all together, and optionally writes
    the result to a file.
    Args:
        input_files: List of paths to TRC JSON files
        output_file: Optional path to write the merged TRC. If None, only returns the result.
        skip_validation: If True, skip validation of the merged TRC
    Returns:
        Merged TRC object
    Raises:
        ValueError: If no TRCs are found or validation fails
        FileNotFoundError: If input files don't exist
    """
    trcs: List[TRC] = []

    # Load all TRCs from input files
    for input_file in input_files:
        file_path = Path(input_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Use from_json to load TRC(s)
        loaded = from_json(file_path)
        if isinstance(loaded, list):
            trcs.extend(loaded)
        else:
            trcs.append(loaded)

    if not trcs:
        raise ValueError("Expected at least one TRC object, found 0")

    # Merge all TRCs using extend
    # Create a deep copy of the first TRC to avoid mutating it
    merged = copy.deepcopy(trcs[0])

    for next_trc in trcs[1:]:
        merged.extend(next_trc)

    # Validate if requested
    if not skip_validation:
        merged.check()

    # Write output if requested
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            f.write(to_json([merged]))

    return merged


def _load_trc(trc: Union[TRC, str, Path]) -> TRC:
    """Load TRC from TRC object or file path."""
    if isinstance(trc, TRC):
        return trc
    elif isinstance(trc, (str, Path)):
        path = Path(trc)
        if not path.exists():
            raise FileNotFoundError(f"TRC file not found: {trc}")
        # Use from_json to load TRC(s)
        loaded = from_json(path)
        if isinstance(loaded, list):
            if len(loaded) == 1:
                return loaded[0]
            else:
                # If multiple TRCs in file, merge them first
                # Create a deep copy of the first TRC to avoid mutating it
                merged = copy.deepcopy(loaded[0])
                for next_trc in loaded[1:]:
                    merged.extend(next_trc)
                return merged
        return loaded
    else:
        raise TypeError(f"TRC must be a TRC object or file path, got {type(trc)}")
