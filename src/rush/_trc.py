"""TRC-related types shared across modules."""

import json
from dataclasses import dataclass
from pathlib import Path

from .client import RushObject, fetch_object
from .convert import from_json
from .mol import TRC


@dataclass(frozen=True)
class TRCPaths:
    """Workspace paths for a saved TRC triplet."""

    topology: Path
    residues: Path
    chains: Path


@dataclass(frozen=True)
class TRCRef:
    """Reference to a single TRC triplet in the Rush object store."""

    topology: RushObject
    residues: RushObject
    chains: RushObject

    def fetch(self) -> TRC:
        """Download and parse into a TRC."""
        return from_json(
            {
                "topology": json.loads(fetch_object(self.topology.path)),
                "residues": json.loads(fetch_object(self.residues.path)),
                "chains": json.loads(fetch_object(self.chains.path)),
            }
        )

    def save(self) -> TRCPaths:
        """Download and save to the workspace."""
        return TRCPaths(
            topology=self.topology.save(),
            residues=self.residues.save(),
            chains=self.chains.save(),
        )
