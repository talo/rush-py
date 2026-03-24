"""TRC-related types shared across modules."""

import json
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import Any

from .client import RushObject, fetch_object, upload_object
from .convert import from_json
from .mol import TRC, Chains, Residues, Topology


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

    @classmethod
    def upload(
        cls,
        trc: TRC,
    ) -> "TRCRef":
        return cls(
            RushObject.from_dict(upload_object(trc.topology.to_dict())),
            RushObject.from_dict(upload_object(trc.residues.to_dict())),
            RushObject.from_dict(upload_object(trc.chains.to_dict())),
        )

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


@singledispatch
def to_topology_vobj(item) -> dict[str, Any]:
    raise NotImplementedError(f"Cannot convert {type(item)} to a Topology!")


@to_topology_vobj.register
def _(trc: TRC) -> dict[str, Any]:
    return upload_object(trc.topology.to_dict())


@to_topology_vobj.register
def _(trc_ref: TRCRef) -> dict[str, Any]:
    return trc_ref.topology.to_dict()


@to_topology_vobj.register
def _(path: Path | str) -> dict[str, Any]:
    return upload_object(path)


@to_topology_vobj.register
def _(object: RushObject) -> dict[str, Any]:
    return object.to_dict()


@to_topology_vobj.register
def _(t: Topology) -> dict[str, Any]:
    return upload_object(t.to_dict())


@singledispatch
def to_residues_vobj(item) -> dict[str, Any]:
    raise NotImplementedError(f"Cannot convert {type(item)} to a residues!")


@to_residues_vobj.register
def _(trc: TRC) -> dict[str, Any]:
    return upload_object(trc.residues.to_dict())


@to_residues_vobj.register
def _(trc_ref: TRCRef) -> dict[str, Any]:
    return trc_ref.residues.to_dict()


@to_residues_vobj.register
def _(path: Path | str) -> dict[str, Any]:
    return upload_object(path)


@to_residues_vobj.register
def _(object: RushObject) -> dict[str, Any]:
    return object.to_dict()


@to_residues_vobj.register
def _(r: Residues) -> dict[str, Any]:
    return upload_object(r.to_dict())


@singledispatch
def to_chains_vobj(item) -> dict[str, Any]:
    raise NotImplementedError(f"Cannot convert {type(item)} to a chains!")


@to_chains_vobj.register
def _(trc: TRC) -> dict[str, Any]:
    return upload_object(trc.chains.to_dict())


@to_chains_vobj.register
def _(trc_ref: TRCRef) -> dict[str, Any]:
    return trc_ref.chains.to_dict()


@to_chains_vobj.register
def _(path: Path | str) -> dict[str, Any]:
    return upload_object(path)


@to_chains_vobj.register
def _(object: RushObject) -> dict[str, Any]:
    return object.to_dict()


@to_chains_vobj.register
def _(c: Chains) -> dict[str, Any]:
    return upload_object(c.to_dict())
