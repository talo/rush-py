"""
Plot FRIED interaction-energy decompositions.

Primary entrypoint:
- plot_fried_stacked(proj_dir, system_prefix, num_top_entries=8, ylim=None) -> Path
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.use("Agg")

__all__ = [
    "plot_fried_stacked",
    "load_json",
    "normalize_residue_block",
    "determine_ligand_fragments",
    "build_fragment_maps",
    "find_reference_outputs",
    "get_residue_name_overrides",
    "build_fragment_key",
    "compute_delta_energy",
    "collect_exess_energies",
]


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def normalize_residue_block(conf: dict):
    """Return (residue_atom_lists, residue_names, residue_ids)."""
    residues_block = conf.get("residues", {})

    if isinstance(residues_block, dict):
        residues = residues_block.get("residues", [])
        seqs = residues_block.get("seqs", [])
        seq_ids = residues_block.get("seq_ns", [])
    else:
        residues = residues_block or []
        aa_seq = conf.get("amino_acid_seq", [])
        residue_seq = conf.get("residue_seq", [])
        seqs = list(aa_seq) + list(residue_seq)

        aa_seq_ids = conf.get("amino_acid_seq_ids", list(range(len(aa_seq))))
        residue_seq_ids = conf.get(
            "residue_seq_ids",
            list(range(len(aa_seq), len(aa_seq) + len(residue_seq))),
        )
        seq_ids = list(aa_seq_ids) + list(residue_seq_ids)

    if not seqs:
        seqs = ["UNK"] * len(residues)
    elif len(sqs := seqs) < len(residues):  # noqa: F841
        seqs = seqs + ["UNK"] * (len(residues) - len(seqs))
    else:
        seqs = seqs[: len(residues)]

    if not seq_ids:
        seq_ids = [None] * len(residues)
    elif len(seq_ids) < len(residues):
        seq_ids = seq_ids + [None] * (len(residues) - len(seq_ids))
    else:
        seq_ids = seq_ids[: len(residues)]

    residue_ids = []
    seen_ids = set()
    for idx, raw_id in enumerate(seq_ids):
        candidate = raw_id
        if candidate in (None, ""):
            candidate = f"r{idx}"
        else:
            candidate = str(candidate)

        if candidate in seen_ids:
            candidate = f"{candidate}_{idx}"
        seen_ids.add(candidate)
        residue_ids.append(candidate)

    return residues, seqs, residue_ids


def determine_ligand_fragments(fragments, ligand_atoms):
    ligand_set = set(ligand_atoms)
    ligand_frags = [
        idx
        for idx, fragment in enumerate(fragments)
        if set(fragment).issubset(ligand_set)
    ]
    return sorted(ligand_frags)


def build_fragment_maps(conf: dict, residue_overrides: dict[str, str] | None = None):
    fragments = conf["topology"].get("fragments", [])
    if not fragments:
        raise ValueError("Topology.fragments is required for plotting.")

    residues, residue_names, residue_ids = normalize_residue_block(conf)
    if not residues:
        raise ValueError("Unable to locate residues information in topology.")

    ligand_atoms = residues[-1]
    ligand_frags = determine_ligand_fragments(fragments, ligand_atoms)
    lig_offset = min(ligand_frags) if ligand_frags else len(fragments)

    frag_to_residue_id = {str(i): str(i) for i in range(len(fragments))}
    for idx in range(min(len(residue_ids), len(fragments))):
        frag_to_residue_id[str(idx)] = residue_ids[idx]

    residue_id_to_name = OrderedDict()
    residue_overrides = residue_overrides or {}
    for res_id, name in zip(residue_ids, residue_names):
        residue_id_to_name[res_id] = residue_overrides.get(res_id, name)

    return frag_to_residue_id, residue_id_to_name, lig_offset


def find_reference_outputs(input_path: Path):
    prefix = input_path.stem
    pattern = f"{prefix}_ref*.json"
    return sorted(input_path.parent.glob(pattern))


def get_residue_name_overrides(input_path: Path) -> dict[str, str]:
    """
    Apply ad-hoc residue naming fixes for known mutants.
    """
    stem = input_path.stem.lower()
    overrides: dict[str, str] = {}
    if stem.startswith("8fln"):
        # 8FLN contains an engineered mutation where residue 481 should be SER.
        overrides["481"] = "SER"
    return overrides


def build_fragment_key(fragment_indices, frag_to_residue_id, lig_offset):
    labels = []
    for frag_idx in sorted(fragment_indices):
        if frag_idx >= lig_offset:
            labels.append(f"f{frag_idx - lig_offset}")
        else:
            labels.append(frag_to_residue_id.get(str(frag_idx), str(frag_idx)))
    return ":".join(labels)


def compute_delta_energy(mer: dict):
    if mer.get("delta_hf_energy") is None:
        return None
    delta_mp2_ss = (
        mer.get("mp2_ss_correction") or mer.get("delta_mp2_ss_correction", 0.0) or 0.0
    )
    delta_mp2_os = (
        mer.get("mp2_os_correction") or mer.get("delta_mp2_os_correction", 0.0) or 0.0
    )
    return mer["delta_hf_energy"] + 0.33 * delta_mp2_ss + 1.2 * delta_mp2_os


def collect_exess_energies(reference_files, frag_to_residue_id, lig_offset):
    exess_energies = {}
    reference_fragments = []

    for ref_file in reference_files:
        result = load_json(ref_file)
        qmmbe = result.get("qmmbe", {})
        ref_frag = qmmbe.get("reference_fragment")
        if ref_frag is not None:
            mapped_id = frag_to_residue_id.get(str(ref_frag))
            if mapped_id:
                reference_fragments.append(mapped_id)

        for mer_group in qmmbe.get("nmers", []):
            for mer in mer_group:
                fragments = mer.get("fragments", [])
                if len(fragments) <= 1:
                    continue

                delta_energy = compute_delta_energy(mer)
                if delta_energy is None:
                    continue

                key = build_fragment_key(
                    [int(idx) for idx in fragments],
                    frag_to_residue_id,
                    lig_offset,
                )
                prev = exess_energies.get(key)
                if prev is not None and abs(prev - delta_energy) >= 1e-6:
                    raise ValueError(
                        f"Inconsistent energy for {key}: {prev} vs {delta_energy}"
                    )
                exess_energies[key] = delta_energy

    return exess_energies, list(dict.fromkeys(reference_fragments))


def key_contains_ligand(key_parts):
    return any(part.startswith("f") for part in key_parts)


def get_keyaa_and_ied(args, input_file: Path):
    conf = load_json(input_file)
    residue_overrides = get_residue_name_overrides(input_file)
    frag_to_residue_id, residue_id_to_name, lig_offset = build_fragment_maps(
        conf, residue_overrides
    )

    reference_files = find_reference_outputs(input_file)
    if not reference_files:
        raise FileNotFoundError(f"No *_ref*.json files found for {input_file.name}")

    exess_energies, aa_ids = collect_exess_energies(
        reference_files, frag_to_residue_id, lig_offset
    )
    if not exess_energies:
        raise ValueError(f"No interaction energies found for {input_file.name}")

    contributions = {}
    total_ied = {}

    for aa_id in aa_ids:
        aa_contrib = {}
        for key, value in exess_energies.items():
            parts = key.split(":")
            if aa_id in parts and key_contains_ligand(parts):
                aa_contrib[key] = value
        if not aa_contrib:
            continue
        contributions[aa_id] = aa_contrib
        total_ied[aa_id] = sum(aa_contrib.values())

    if not contributions:
        raise ValueError(f"No AA-ligand contributions computed for {input_file.name}")

    top_ied = sorted(
        total_ied, key=lambda residue: abs(total_ied[residue]), reverse=True
    )[: args.num_top_entries]

    key_amino_acids = OrderedDict()
    for aa_id in top_ied:
        label = residue_id_to_name.get(aa_id, "RES")
        key_amino_acids[aa_id] = f"{label}_{aa_id}"

    return key_amino_acids, contributions, total_ied, residue_id_to_name


def plot_stacked(args, key_amino_acids, contributions, total_ied):
    if not key_amino_acids:
        print(f"No key amino acids to plot for {args.system_prefix}")
        return

    fig, ax = plt.subplots(figsize=(max(len(key_amino_acids), 1) * 1.2, 12))

    contributions = {
        x_val: contributions[x_val]
        for x_val in key_amino_acids
        if x_val in contributions
    }
    total_ied = {
        x_val: total_ied[x_val] for x_val in key_amino_acids if x_val in total_ied
    }

    bar_width = 0.7
    threshold = 0.3
    colormap = {
        "f0": [0.98, 0.60, 0.60],
        "f1": [0.70, 0.80, 0.89],
        "f2": [0.80, 0.92, 0.77],
        "f3": [0.99, 0.88, 0.71],
        "f4": [0.89, 0.78, 0.89],
        "f5": [1.00, 1.00, 0.80],
    }

    for idx, x_val in enumerate(key_amino_acids):
        if x_val not in contributions:
            ax.bar(idx, 0, width=bar_width, color="grey", alpha=0.5)
            continue

        contribution = contributions[x_val]
        try:
            labels, values = zip(
                *sorted(
                    contribution.items(),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )
            )
        except ValueError:
            ax.bar(idx, 0, width=bar_width, color="grey", alpha=0.5)
            continue

        values = [value * 627.509 for value in values]
        total_ied[x_val] *= 627.509

        positive_cumulative = sum(value for value in values if value >= 0)
        negative_cumulative = sum(value for value in values if value < 0)

        plot_positive_cumulative = 0
        plot_negative_cumulative = 0
        plot_positive_contributor = True
        plot_negative_contributor = True

        for label, value in zip(labels, values):
            color_keys = label.split(":")[1:]
            colors = [colormap.get(key, [0.5, 0.5, 0.5]) for key in color_keys]
            color = [sum(color[i] for color in colors) / len(colors) for i in range(3)]

            if value >= 0:
                bottom = plot_positive_cumulative
                plot_positive_cumulative += value
                should_plot = plot_positive_contributor
            else:
                bottom = plot_negative_cumulative
                plot_negative_cumulative += value
                should_plot = plot_negative_contributor

            if should_plot:
                ax.bar(
                    idx,
                    value,
                    bottom=bottom,
                    width=bar_width,
                    color=color,
                    alpha=0.8,
                )
                ax.text(
                    idx,
                    bottom + value / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

                if abs(value) < threshold:
                    plot_positive_contributor = False
                    plot_negative_contributor = False
        else:
            ax.bar(idx, 0, width=bar_width, color="grey", alpha=0.5)

    total_values = [
        total_ied[x_val] if x_val in total_ied else None for x_val in key_amino_acids
    ]
    valid_indices = [idx for idx, value in enumerate(total_values) if value is not None]
    valid_values = [value for value in total_values if value is not None]

    if valid_indices:
        ax.plot(
            valid_indices,
            valid_values,
            color="grey",
            linestyle="None",
            marker="o",
            markersize=8,
            label="Total AA-ligand Interaction Energy",
        )

    if args.ylim:
        ax.set_ylim([float(val) for val in args.ylim.split(",")[::-1]])

    ax.set_xticks(range(len(key_amino_acids)))
    ax.set_xticklabels([key_amino_acids[key] for key in key_amino_acids])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Contribution Value / Energy (Kcal/mol)")
    ax.set_xlabel("AA ID")
    ax.set_title(
        f"Top Fragment Contributions with Key Amino Acids: {args.system_prefix}"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"fried_{args.system_prefix}.png")
    plt.close(fig)


def plot_fried_stacked(
    proj_dir: str | Path,
    system_prefix: str,
    num_top_entries: int = 8,
    ylim: str | None = None,
) -> Path:
    """
    Plot FRIED interaction energy contributions for a given system prefix.
    """
    proj_dir = Path(proj_dir)
    fraglig_file = proj_dir / f"{system_prefix}_fraglig.json"

    if not fraglig_file.exists():
        raise FileNotFoundError(f"Fraglig file not found: {fraglig_file}")

    class Args:
        def __init__(self):
            self.num_top_entries = num_top_entries
            self.ylim = ylim
            self.system_prefix = system_prefix

    args = Args()
    (key_amino_acids, contributions, total_ied, residue_id_to_name) = get_keyaa_and_ied(
        args, fraglig_file
    )
    system_labels = OrderedDict(
        (res_id, f"{residue_id_to_name.get(res_id, 'RES')}_{res_id}")
        for res_id in key_amino_acids
    )
    plot_stacked(args, system_labels, contributions, total_ied)

    return Path(f"{system_prefix}.png")
