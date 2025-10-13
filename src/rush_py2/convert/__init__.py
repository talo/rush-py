"""
Conversion utilities for molecular structure file formats.

This module provides functions to convert between PDB, mmCIF, and JSON formats.
"""

from typing import List, Optional

from .pdb import from_pdb, to_pdb
from .mmcif import from_mmcif
from .json import from_json, to_json
from ..mol import TRC


def load_structure(file_path: str) -> List[TRC]:
    """
    Load structure from PDB, mmCIF, or JSON file.
    
    Args:
        file_path: Path to structure file
        
    Returns:
        List of TRC structures
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Determine file type by extension
    if file_path.lower().endswith('.json'):
        return from_json(content)
    elif file_path.lower().endswith('.cif'):
        return from_mmcif(content)
    elif file_path.lower().endswith('.pdb'):
        return from_pdb(content)
    else:
        # Try to guess from content
        content_lower = content.lower()
        if content.strip().startswith('[') or content.strip().startswith('{'):
            return from_json(content)
        elif 'data_' in content_lower and '_atom_site' in content_lower:
            return from_mmcif(content)
        else:
            return from_pdb(content)


def save_structure(trcs: List[TRC], file_path: str, format: Optional[str] = None):
    """
    Save TRC structures to file.
    
    Args:
        trcs: List of TRC structures
        file_path: Output file path
        format: Output format ('pdb', 'json', or None for auto-detect from extension)
    """
    if format is None:
        # Auto-detect from extension
        if file_path.lower().endswith('.json'):
            format = 'json'
        elif file_path.lower().endswith('.pdb'):
            format = 'pdb'
        else:
            format = 'pdb'  # Default
    
    if format.lower() == 'json':
        content = to_json(trcs)
    elif format.lower() == 'pdb':
        if len(trcs) > 1:
            # Multi-model PDB
            content_parts = []
            for i, trc in enumerate(trcs, 1):
                content_parts.append(f"MODEL     {i:>4}")
                content_parts.append(to_pdb(trc).replace("END\n", ""))
                content_parts.append("ENDMDL")
            content_parts.append("END")
            content = '\n'.join(content_parts)
        else:
            content = to_pdb(trcs[0])
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    with open(file_path, 'w') as f:
        f.write(content)


__all__ = [
    'from_pdb',
    'to_pdb',
    'from_mmcif',
    'from_json',
    'to_json',
    'load_structure',
    'save_structure',
]

