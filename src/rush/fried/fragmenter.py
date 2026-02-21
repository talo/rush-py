"""
Ligand fragmentation utilities.

Primary entrypoint:
- fragment_ligand(input_path, fragment_assignments_lookup=None) -> Path
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[unresolved-import]

__all__ = [
    "fragment_ligand",
    "parse_graph",
    "detect_protected_edges",
    "fragment_graph",
    "integrate_fragments",
]


def parse_graph(conf: dict[str, Any]) -> nx.Graph:
    """Build a molecular graph for the ligand from a complex JSON."""
    mol_graph = nx.Graph()

    ligand = conf["residues"]["residues"][-1]

    for idx in ligand:
        mol_graph.add_node(idx, label=conf["topology"]["symbols"][idx])

    for origin, dest, order in conf["topology"]["connectivity"]:
        if origin in ligand:
            mol_graph.add_edge(origin, dest, order=order)
    return mol_graph


def detect_protected_edges(
    mol_graph: nx.Graph, fragment_assignments: list[list[int]] | None
) -> set[tuple[int, int]]:
    """Return edges that should not be cut during fragmentation."""
    cycles = nx.cycle_basis(mol_graph)
    protected_edges: set[tuple[int, int]] = set()

    # Protect all edges connected to H, multiple bonds
    for u, v, data in mol_graph.edges(data=True):
        label_u = mol_graph.nodes[u]["label"]
        label_v = mol_graph.nodes[v]["label"]
        if label_u == "H" or label_v == "H" or data["order"] > 1:
            protected_edges.add((u, v))

    # Protect all edges in cycles
    for cycle in cycles:
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            protected_edges.add((u, v))

    if fragment_assignments:
        for u, v, _ in mol_graph.edges(data=True):
            for frag in fragment_assignments:
                if u in frag and v in frag:
                    protected_edges.add((u, v))

    return protected_edges


def integrate_fragments(
    graph: nx.Graph, fragments: list[list[int]], min_fragment_size: int = 5
) -> list[list[int]]:
    """
    Integrate small fragments into larger ones based on connectivity and chemistry.
    """
    checked_fragments: list[list[int]] = []
    small_fragments: list[list[int]] = []
    for frag in fragments:
        (
            small_fragments if len(frag) < min_fragment_size else checked_fragments
        ).append(frag)
    while small_fragments:
        small_fragment = small_fragments.pop(0)
        if len(small_fragment) > 2:
            subgraph = graph.subgraph(small_fragment)
            if any(order == 2 for _, _, order in subgraph.edges(data="order")):
                checked_fragments.append(small_fragment)
                continue
        merged = False
        for fragment in checked_fragments:
            if any(
                neighbor in fragment
                for node in small_fragment
                for neighbor in graph.neighbors(node)
            ):
                fragment.extend(n for n in small_fragment if n not in fragment)
                merged = True
                break
        if not merged:
            small_fragments.append(small_fragment)
        else:
            checked_fragments = [sorted(frag) for frag in checked_fragments]
    return checked_fragments


def fragment_graph(
    mol_graph: nx.Graph, protected_edges: set[tuple[int, int]]
) -> list[list[int]]:
    """Fragment the graph by cutting eligible single bonds, then integrate small fragments."""
    original_graph = mol_graph.copy()

    for u, v, data in list(mol_graph.edges(data=True)):
        if (
            data["order"] == 1
            and (u, v) not in protected_edges
            and (v, u) not in protected_edges
        ):
            mol_graph.remove_edge(u, v)
    fragments = integrate_fragments(
        original_graph, [list(comp) for comp in nx.connected_components(mol_graph)]
    )
    return fragments


def fragment_ligand(
    input_path: str | Path, fragment_assignments_lookup: dict[str, Any] | None = None
) -> Path:
    """
    Fragment a ligand in a complex JSON file and write a *_fraglig.json alongside it.
    """
    input_path = Path(input_path)
    with open(input_path) as f:
        conf = json.load(f)

    fragment_assignments = (fragment_assignments_lookup or {}).get(input_path.stem, [])

    mol_graph = parse_graph(conf)
    protected_edges = detect_protected_edges(mol_graph, fragment_assignments)
    fragments = fragment_graph(mol_graph, protected_edges)

    flat_fragments = [f for fragment in fragments for f in fragment]
    if len(flat_fragments) != len(set(flat_fragments)):
        raise ValueError("Duplicate fragment index detected after fragmentation.")
    missing_fragments = [
        f for f in flat_fragments if f not in conf["residues"]["residues"][-1]
    ]
    if missing_fragments:
        raise ValueError(f"Missing fragment index(es): {missing_fragments}")

    conf["topology"]["fragments"] = conf["topology"]["fragments"][:-1] + fragments
    conf["topology"]["fragment_formal_charges"] = conf["topology"][
        "fragment_formal_charges"
    ] + [0] * (len(fragments) - 1)

    # Validate fragment index continuity
    all_indices = sorted(
        [index for fragment in conf["topology"]["fragments"] for index in fragment]
    )
    for i, frag_id in enumerate(all_indices):
        if i != frag_id:
            raise ValueError(f"fragment_index_error: {i} != {frag_id}")

    if len(conf["topology"]["fragments"]) != len(
        conf["topology"]["fragment_formal_charges"]
    ):
        raise ValueError(
            "len(fragments) != len(fragment_formal_charges) after fragmentation"
        )

    output_path = input_path.with_name(f"{input_path.stem}_fraglig.json")
    with open(output_path, "w") as f:
        json.dump(conf, f, indent=2)
    return output_path
