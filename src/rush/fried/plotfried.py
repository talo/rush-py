"""
Plot FRIED interaction-energy decompositions.

Primary entrypoint:
- plot_fried_stacked(<proj_dir>, <system_prefix>, num_top_entries=8, ylim=None, colormap="tab20")
"""

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARTREE_TO_KCAL = 627.509


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
    elif len(seqs) < len(residues):
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


def build_fragment_maps(conf: dict):
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
    for res_id, name in zip(residue_ids, residue_names):
        residue_id_to_name[res_id] = name

    return frag_to_residue_id, residue_id_to_name, lig_offset


def find_reference_outputs(input_path: Path):
    prefix = input_path.stem
    pattern = f"{prefix}_ref*.json"
    return sorted(input_path.parent.glob(pattern))


def _fragment_label_to_color(label: str, cmap) -> tuple[float, float, float]:
    """Map a fragment label like 'f3' onto a color from the provided colormap."""
    if not label.startswith("f"):
        return (0.6, 0.6, 0.6)
    try:
        idx = int(label[1:])
    except ValueError:
        return (0.6, 0.6, 0.6)

    rgba = cmap(idx % cmap.N)
    return tuple(rgba[:3])


def _average_colors(
    colors: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Average multiple RGB tuples; fall back to grey if the list is empty."""
    if not colors:
        return (0.5, 0.5, 0.5)
    color_count = len(colors)
    return (
        sum(color[0] for color in colors) / color_count,
        sum(color[1] for color in colors) / color_count,
        sum(color[2] for color in colors) / color_count,
    )


def build_contribution_report(
    system_labels: OrderedDict, contributions: dict, total_ied: dict
) -> OrderedDict:
    """Structure contribution data and convert to kcal/mol for downstream use."""
    report = OrderedDict()
    for aa_id, label in system_labels.items():
        aa_contrib = contributions.get(aa_id)
        if not aa_contrib:
            continue
        sorted_entries = sorted(
            aa_contrib.items(), key=lambda item: abs(item[1]), reverse=True
        )
        report[aa_id] = {
            "label": label,
            "total_energy_hartree": total_ied.get(aa_id, 0.0),
            "total_energy_kcal_per_mol": total_ied.get(aa_id, 0.0) * HARTREE_TO_KCAL,
            "contributions": [
                {
                    "key": key,
                    "energy_hartree": value,
                    "energy_kcal_per_mol": value * HARTREE_TO_KCAL,
                }
                for key, value in sorted_entries
            ],
        }
    return report


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
    delta_mp2_ss = mer.get("delta_mp2_ss_correction", 0.0) or 0.0
    delta_mp2_os = mer.get("delta_mp2_os_correction", 0.0) or 0.0
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
    frag_to_residue_id, residue_id_to_name, lig_offset = build_fragment_maps(conf)

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


def plot_stacked(
    args, report_data: OrderedDict[str, dict], output_dir: Path
) -> Path | None:
    if not report_data:
        print(f"No key amino acids to plot for {args.system_prefix}")
        return None

    fig, ax = plt.subplots(figsize=(max(len(report_data), 1) * 1.2, 12))

    bar_width = 0.7
    threshold = 0.3
    cmap_name = getattr(args, "colormap", "tab20")
    try:
        fragment_cmap = plt.get_cmap(cmap_name)
    except ValueError:
        fragment_cmap = plt.get_cmap("tab20")

    for idx, (aa_id, entry) in enumerate(report_data.items()):
        aa_contrib = entry["contributions"]
        if not aa_contrib:
            ax.bar(idx, 0, width=bar_width, color="grey", alpha=0.5)
            continue

        labels = [item["key"] for item in aa_contrib]
        values = [item["energy_kcal_per_mol"] for item in aa_contrib]

        _positive_cumulative = sum(value for value in values if value >= 0)
        _negative_cumulative = sum(value for value in values if value < 0)

        plot_positive_cumulative = 0
        plot_negative_cumulative = 0
        plot_positive_contributor = True
        plot_negative_contributor = True

        for label, value in zip(labels, values):
            color_keys = [key for key in label.split(":")[1:] if key.startswith("f")]
            colors = [
                _fragment_label_to_color(key, fragment_cmap) for key in color_keys
            ]
            color = _average_colors(colors)

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
        entry["total_energy_kcal_per_mol"] for entry in report_data.values()
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
        v = [float(val) for val in args.ylim.split(",")[::-1]]
        ax.set_ylim((v[0], v[1]))

    xtick_labels = [entry["label"] for entry in report_data.values()]
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Contribution Value / Energy (kcal/mol)")
    ax.set_xlabel("AA ID")
    ax.set_title(
        f"Top Fragment Contributions with Key Amino Acids: {args.system_prefix}"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fried_{args.system_prefix}.png"
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_fried_stacked(
    proj_dir: str | Path,
    system_prefix: str,
    num_top_entries: int = 8,
    ylim: str | None = None,
    colormap: str = "tab20",
) -> Path:
    """
    Plot FRIED interaction energy contributions.

    Args:
        proj_dir: Project directory containing fraglig.json and *_ref*.json files
        system_prefix: System prefix (e.g., "3fly" for 3fly_fraglig.json)
        num_top_entries: Number of top contributors to plot
        ylim: Optional y-axis limits (e.g., "-30,10")
        colormap: Matplotlib colormap name used for ligand fragment colors

    Returns:
        Path to the saved plot file
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
            self.colormap = colormap

    args = Args()
    (
        key_amino_acids,
        contributions,
        total_ied,
        residue_id_to_name,
    ) = get_keyaa_and_ied(args, fraglig_file)
    # Rebuild labels using the current system's residue-name map.
    system_labels = OrderedDict(
        (res_id, f"{residue_id_to_name.get(res_id, 'RES')}_{res_id}")
        for res_id in key_amino_acids
    )
    report_data = build_contribution_report(system_labels, contributions, total_ied)
    output_path = plot_stacked(args, report_data, proj_dir)

    summary_path = proj_dir / f"fried_{system_prefix}.json"
    with summary_path.open("w") as handle:
        json.dump(contributions, handle, indent=2)

    if output_path:
        print(f"[plot] Saved stacked bar chart to {output_path}")
    print(f"[plot] Saved contribution summary to {summary_path}")

    return output_path or summary_path
