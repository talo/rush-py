"""
JSON conversion functionality for TRC structures.
"""

import json
from pathlib import Path

from ..mol import TRC, Chains, Residues, Topology


def from_json(
    json_content: (
        str | Path | tuple[str | Path, str | Path, str | Path] | dict | list[dict]
    ),
) -> TRC | list[TRC]:
    """
    Load TRC structures from JSON.

    Args:
        json_content: JSON string content

    Returns:
        TRC structure or list of TRC structures
    """
    if isinstance(json_content, str):
        data = json.loads(json_content)
    elif isinstance(json_content, Path):
        with json_content.open() as f:
            data = json.load(f)
    elif isinstance(json_content, tuple) and len(json_content) == 3:
        data = [{}]
        with (
            Path(json_content[0]).open() as t_f,
            Path(json_content[1]).open() as r_f,
            Path(json_content[2]).open() as c_f,
        ):
            data[0]["topology"] = json.load(t_f)
            data[0]["residues"] = json.load(r_f)
            data[0]["chains"] = json.load(c_f)
    elif isinstance(json_content, dict):
        data = [json_content]
    else:
        data = json_content

    # Turn single TRCs into lists
    if isinstance(data, dict) and "topology" in data:
        data = [data]

    trcs = []
    for trc_data in data:
        # Load topology, residues, and chains
        topology = Topology.from_json(trc_data["topology"])
        residues = Residues.from_json(trc_data["residues"])
        chains = Chains.from_json(trc_data["chains"])

        # Create TRC
        trc = TRC(topology=topology, residues=residues, chains=chains)
        trcs.append(trc)

    if len(trcs) == 1:
        return trcs[0]
    else:
        return trcs


def to_json(trcs: TRC | list[TRC]) -> str:
    """
    Convert TRC structures to JSON.

    Args:
        trcs: TRC structure or list of TRC structures

    Returns:
        JSON string
    """
    data = []

    if isinstance(trcs, TRC):
        trcs = [trcs]
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
                "labeled": (
                    [r.value for r in trc.residues.labeled]
                    if trc.residues.labeled is not None
                    else None
                ),
                "labels": (
                    trc.residues.labels if trc.residues.labels is not None else None
                ),
            },
            "chains": {
                "chains": [chain.residues for chain in trc.chains.chains],
                "alpha_helices": (
                    [r.value for r in trc.chains.alpha_helices]
                    if trc.chains.alpha_helices
                    else None
                ),
                "beta_sheets": (
                    [r.value for r in trc.chains.beta_sheets]
                    if trc.chains.beta_sheets
                    else None
                ),
                "labeled": (
                    [r.value for r in trc.chains.labeled]
                    if trc.chains.labeled
                    else None
                ),
                "labels": (
                    trc.chains.labels if trc.chains.labels is not None else None
                ),
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
