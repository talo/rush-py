from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TRCSavedResult:
    """Workspace paths for a saved TRC triplet."""

    topology: Path
    residues: Path
    chains: Path
