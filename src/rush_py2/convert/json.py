"""
JSON conversion functionality for TRC structures.
"""

import json
from typing import List

from ..mol import (
    TRC,
    Chain,
    Chains,
    Element,
    FormalCharge,
    Fragment,
    PartialCharge,
    Residue,
    ResidueRef,
    Residues,
    SchemaVersion,
    Topology,
)


def from_json(json_content: str) -> List[TRC]:
    """
    Load TRC structures from JSON.

    Args:
        json_content: JSON string content

    Returns:
        List of TRC structures
    """
    data = json.loads(json_content)
    trcs = []

    for trc_data in data:
        # Load topology
        topology_data = trc_data["topology"]
        topology = Topology()
        topology.schema_version = (
            SchemaVersion.V2
        )  # Default, could parse from schema_version
        topology.symbols = [Element.from_str(s) for s in topology_data["symbols"]]
        topology.geometry = topology_data["geometry"]

        if "labels" in topology_data and topology_data["labels"]:
            topology.labels = topology_data["labels"]

        if "formal_charges" in topology_data and topology_data["formal_charges"]:
            topology.formal_charges = [
                FormalCharge(c) for c in topology_data["formal_charges"]
            ]

        if "partial_charges" in topology_data and topology_data["partial_charges"]:
            topology.partial_charges = [
                PartialCharge(c) for c in topology_data["partial_charges"]
            ]

        if "velocities" in topology_data and topology_data["velocities"]:
            topology.velocities = topology_data["velocities"]

        if "connectivity" in topology_data and topology_data["connectivity"]:
            # Connectivity format needs to be determined from actual JSON
            pass

        if "fragments" in topology_data and topology_data["fragments"]:
            topology.fragments = [Fragment(frag) for frag in topology_data["fragments"]]

        # Load residues
        residues_data = trc_data["residues"]
        residues = Residues()
        residues.residues = [Residue(res) for res in residues_data["residues"]]
        residues.seqs = residues_data["seqs"]
        residues.seq_ns = residues_data["seq_ns"]
        residues.insertion_codes = residues_data["insertion_codes"]

        # Load chains
        chains_data = trc_data["chains"]
        chains = Chains()
        chains.chains = [Chain(chain) for chain in chains_data["chains"]]

        if chains_data.get("alpha_helices"):
            chains.alpha_helices = [ResidueRef(r) for r in chains_data["alpha_helices"]]

        if chains_data.get("beta_sheets"):
            chains.beta_sheets = [ResidueRef(r) for r in chains_data["beta_sheets"]]

        # Create TRC
        trc = TRC(topology=topology, residues=residues, chains=chains)
        trcs.append(trc)

    return trcs


def to_json(trcs: List[TRC]) -> str:
    """
    Convert TRC structures to JSON.

    Args:
        trcs: List of TRC structures

    Returns:
        JSON string
    """
    data = []

    for trc in trcs:
        # Build topology dict with only the fields that exist in expected format
        topology_dict = {
            "schema_version": "0.2.0",
            "symbols": [str(symbol) for symbol in trc.topology.symbols],
            "geometry": trc.topology.geometry,
        }

        # Add optional fields only if they have data or are expected to be null
        if trc.topology.labels:
            topology_dict["labels"] = trc.topology.labels

        if trc.topology.formal_charges:
            topology_dict["formal_charges"] = [
                c.charge for c in trc.topology.formal_charges
            ]

        # Always include connectivity and fragments as empty lists if not present (based on expected JSON)
        topology_dict["connectivity"] = []
        topology_dict["fragments"] = []
        topology_dict["fragment_formal_charges"] = []

        trc_data = {
            "topology": topology_dict,
            "residues": {
                "residues": [residue.atoms for residue in trc.residues.residues],
                "seqs": trc.residues.seqs,
                "seq_ns": trc.residues.seq_ns,
                "insertion_codes": trc.residues.insertion_codes,
                "labeled": None,
                "labels": None,
            },
            "chains": {
                "chains": [chain.residues for chain in trc.chains.chains],
                "alpha_helices": [r.value for r in trc.chains.alpha_helices]
                if trc.chains.alpha_helices
                else None,
                "beta_sheets": [r.value for r in trc.chains.beta_sheets]
                if trc.chains.beta_sheets
                else None,
                "labeled": [r.value for r in trc.chains.labeled]
                if trc.chains.labeled
                else None,
                "labels": trc.chains.labels,
            },
        }

        # Set connectivity if exists
        if trc.topology.connectivity:
            topology_dict["connectivity"] = [
                [bond.atom1.value, bond.atom2.value, bond.order.value]
                for bond in trc.topology.connectivity
            ]

        # Set fragments if exists
        if trc.topology.fragments:
            topology_dict["fragments"] = [
                fragment.atoms for fragment in trc.topology.fragments
            ]

        # Set fragment formal charges if exists
        if trc.topology.fragment_formal_charges:
            topology_dict["fragment_formal_charges"] = [
                c.charge for c in trc.topology.fragment_formal_charges
            ]

        data.append(trc_data)

    return json.dumps(data, indent=2)
