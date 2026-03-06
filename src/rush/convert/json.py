"""
JSON conversion functionality for TRC structures.
"""

import json
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import TypeGuard, overload

from ..mol import TRC, Chains, Residues, Topology

StrPath = PathLike[str]


def _trcs_from_dicts(trc_dicts: list[dict]) -> list[TRC]:
    return [
        TRC(
            topology=Topology.from_json(trc_dict["topology"]),
            residues=Residues.from_json(trc_dict["residues"]),
            chains=Chains.from_json(trc_dict["chains"]),
        )
        for trc_dict in trc_dicts
    ]


def _is_dict_list(value: object) -> TypeGuard[list[dict]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _is_path_triplet(value: object) -> TypeGuard[Sequence[StrPath]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != 3:
        return False
    return all(isinstance(item, PathLike) for item in value)


def _is_str_path(value: object) -> TypeGuard[StrPath]:
    return isinstance(value, PathLike)


def _normalize_trc_json(data: object) -> list[dict]:
    if isinstance(data, dict) and "topology" in data:
        return [data]
    if _is_dict_list(data):
        return data
    raise TypeError("Expected TRC JSON dict or list of dicts")


@overload
def from_json(json_content: Sequence[StrPath] | dict) -> TRC: ...


@overload
def from_json(json_content: StrPath | list[dict]) -> list[TRC]: ...


def from_json(
    json_content: Sequence[StrPath] | dict | StrPath | list[dict],
) -> TRC | list[TRC]:
    """
    Load TRC structures from JSON.

    Args:
        json_content: JSON file path, dict, list of dicts, or (topology, residues, chains) paths

    Returns:
        TRC structure or list of TRC structures
    """
    if isinstance(json_content, dict):
        trcs = _trcs_from_dicts([json_content])
        return trcs[0]

    if _is_dict_list(json_content):
        return _trcs_from_dicts(json_content)

    if _is_path_triplet(json_content):
        data = {}
        with (
            Path(json_content[0]).open() as t_f,
            Path(json_content[1]).open() as r_f,
            Path(json_content[2]).open() as c_f,
        ):
            data["topology"] = json.load(t_f)
            data["residues"] = json.load(r_f)
            data["chains"] = json.load(c_f)
        trcs = _trcs_from_dicts([data])
        return trcs[0]

    if _is_str_path(json_content):
        path = Path(json_content)
        with path.open() as f:
            data = json.load(f)
        trc_dicts = _normalize_trc_json(data)
        return _trcs_from_dicts(trc_dicts)

    raise TypeError(
        "Expected a path, dict, list of dicts, or (topology, residues, chains) paths"
    )


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
