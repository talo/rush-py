"""
JSON conversion functionality for TRC structures.
"""

import json
from pathlib import Path

from ..mol import TRC, Chains, Residues, Topology


def from_json(
    json_content: (
        str
        | Path
        | tuple[str | Path, str | Path, str | Path]
        | dict
        | list[dict]
        | tuple[Path, ...]
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
        data = {}
        with (
            Path(json_content[0]).open() as t_f,
            Path(json_content[1]).open() as r_f,
            Path(json_content[2]).open() as c_f,
        ):
            data["topology"] = json.load(t_f)
            data["residues"] = json.load(r_f)
            data["chains"] = json.load(c_f)
    elif isinstance(json_content, dict):
        data = [json_content]
    else:
        data = json_content

    # Turn single TRCs into lists
    if isinstance(data, dict) and "topology" in data:
        data = [data]

    trcs = [
        TRC(
            topology=Topology.from_json(trc["topology"]),
            residues=Residues.from_json(trc["residues"]),
            chains=Chains.from_json(trc["chains"]),
        )
        for trc in data
    ]

    return trcs[0] if len(trcs) == 1 else trcs


def to_json(trcs: TRC | list[TRC]) -> dict[str, object] | list[dict[str, object]]:
    """
    Convert TRC structures to JSON.

    Args:
        trcs: TRC structure or list of TRC structures

    Returns:
        JSON-compatible dict or list of dicts
    """

    if isinstance(trcs, TRC):
        trcs = [trcs]

    data = [
        {
            "topology": trc.topology.to_json(),
            "residues": trc.residues.to_json(),
            "chains": trc.chains.to_json(),
        }
        for trc in trcs
    ]

    return data[0] if len(data) == 1 else data
