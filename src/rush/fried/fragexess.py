"""
Submit EXESS interaction-energy jobs for ligand fragments (FRIED workflow).

Primary entrypoint:
- fragmented_exess(input_file, distance_threshold=4.0, trimer_cutoff_cap=15.0, collect=True, output_dir=None) -> None
"""

import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from rush.mol import FragmentRef

from .. import RunError, exess
from ..exess import interaction_energy
from ..runs import Run, RunOpts

__all__ = [
    "fragmented_exess",
    "determine_ligand_atoms",
    "collect_ligand_fragments",
    "compute_fragment_cutoffs",
    "build_frag_keywords",
    "discover_inputs",
]


@dataclass(frozen=True)
class FragmentJob:
    reference_fragment: int
    cutoff: int


def _load_conf(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _materialize_topology(
    conf: dict[str, Any], source_path: Path, staging_dir: Path
) -> Path:
    """
    Extract topology from conformer and write to a staging file with validation.
    """
    topology = conf.get("topology")
    if not topology:
        raise ValueError(f"No 'topology' field found in {source_path}")
    if "schema_version" not in topology:
        raise ValueError(
            "Topology block missing 'schema_version'; cannot submit to EXESS."
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    topo_path = staging_dir / f"{source_path.stem}_topology.json"

    with open(topo_path, "w") as handle:
        json.dump(topology, handle, indent=2)

    if not topo_path.exists() or topo_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to create valid topology file: {topo_path}")

    # Validate JSON
    with open(topo_path) as f:
        json.load(f)

    return topo_path


def _chunk_geometry(
    geometry: Sequence[float], atom_index: int
) -> tuple[float, float, float]:
    start = atom_index * 3
    slice_ = geometry[start : start + 3]
    if len(slice_) != 3:
        raise ValueError(f"Atom {atom_index} missing coordinates in geometry block.")
    return float(slice_[0]), float(slice_[1]), float(slice_[2])


def _distance(coord_a: Sequence[float], coord_b: Sequence[float]) -> float:
    return math.sqrt(sum((ax - bx) ** 2 for ax, bx in zip(coord_a, coord_b)))


def _normalize_labels(label: Any) -> list[str]:
    if isinstance(label, str):
        return [label]
    if isinstance(label, Iterable):
        return [str(item) for item in label]
    return [str(label)]


def _extract_residue_block(
    conf: dict[str, Any],
) -> tuple[list[list[int]], list[int], list[list[str]]]:
    residues_block = conf.get("residues", {})
    if isinstance(residues_block, dict):
        residues = residues_block.get("residues", []) or []
        labeled = residues_block.get("labeled", []) or []
        labels = residues_block.get("labels", []) or []
    else:
        residues = residues_block or []
        labeled = []
        labels = []
    labels = [_normalize_labels(label) for label in labels]
    return residues, labeled, labels


def determine_ligand_atoms(conf: dict[str, Any]) -> tuple[set[int], list[int]]:
    residues, labeled, labels = _extract_residue_block(conf)
    ligand_res_indices: list[int] = [
        idx
        for idx, label_tokens in zip(labeled, labels)
        if any(token.lower().startswith("lig") for token in label_tokens)
    ]

    if not ligand_res_indices and residues:
        ligand_res_indices = [len(residues) - 1]

    ligand_atoms: set[int] = set()
    for idx in ligand_res_indices:
        if idx >= len(residues):
            continue
        ligand_atoms.update(residues[idx])

    if not ligand_atoms:
        raise ValueError("Unable to locate ligand atoms in the provided topology.")

    return ligand_atoms, ligand_res_indices


def collect_ligand_fragments(
    conf: dict[str, Any], ligand_atoms: set[int]
) -> list[FragmentRef]:
    ligand_fragments: list[FragmentRef] = []
    for frag_idx, fragment in enumerate(conf["topology"]["fragments"]):
        if set(fragment) & ligand_atoms:
            ligand_fragments.append(FragmentRef(frag_idx))
    if not ligand_fragments:
        raise ValueError("Failed to match ligand atoms to fragment indices.")
    return ligand_fragments


def compute_fragment_cutoffs(
    conf: dict[str, Any],
    ligand_atoms: set[int],
    ligand_fragments: set[FragmentRef],
    threshold: float,
) -> list[FragmentJob]:
    geometry = conf["topology"]["geometry"]
    ligand_coords = [
        _chunk_geometry(geometry, atom_idx) for atom_idx in sorted(ligand_atoms)
    ]

    fragment_jobs: list[FragmentJob] = []
    for frag_idx, fragment in enumerate(conf["topology"]["fragments"]):
        if frag_idx in ligand_fragments:
            continue
        if not fragment:
            continue
        frag_coords = [_chunk_geometry(geometry, atom_idx) for atom_idx in fragment]
        distances = [
            _distance(coord, lig_coord)
            for coord in frag_coords
            for lig_coord in ligand_coords
        ]
        if not distances:
            continue
        if min(distances) >= threshold:
            continue
        cutoff = int(max(distances)) + 1
        fragment_jobs.append(FragmentJob(reference_fragment=frag_idx, cutoff=cutoff))
    return fragment_jobs


def build_frag_keywords(
    cutoff: int,
    reference_fragment: int,
    included_fragments: list[FragmentRef],
    trimer_cap: float,
) -> exess.FragKeywords:
    trimer_cutoff = float(min(cutoff, trimer_cap))
    included = sorted(set(included_fragments + [reference_fragment]))
    return exess.FragKeywords(
        level="Trimer",
        dimer_cutoff=float(cutoff),
        trimer_cutoff=trimer_cutoff,
        tetramer_cutoff=trimer_cutoff,
        cutoff_type="Centroid",
        included_fragments=included,
    )


def fragmented_exess(
    input_file: str | Path,
    distance_threshold: float = 4.0,
    trimer_cutoff_cap: float = 15.0,
    collect: bool = True,
    output_dir: Path | None = None,
) -> list[Run[exess.ResultRef]] | None:
    """
    Submit EXESS calculations for a fragmented ligand complex.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = output_dir or input_path.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_root = Path(tempfile.gettempdir()) / "tmp_exess_topologies"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / input_path.stem

    conf = _load_conf(input_path)
    topology_path = _materialize_topology(conf, input_path, staging_dir)
    ligand_atoms, _ = determine_ligand_atoms(conf)
    ligand_fragments = collect_ligand_fragments(conf, ligand_atoms)
    fragment_jobs = compute_fragment_cutoffs(
        conf,
        ligand_atoms,
        set(ligand_fragments),
        threshold=distance_threshold,
    )

    if not fragment_jobs:
        print(f"  WARNING: No fragments within {distance_threshold} Å of ligand")
        return

    if not topology_path.exists():
        raise RuntimeError(f"Topology file no longer exists: {topology_path}")

    submitted: list[Run[exess.ResultRef]] = []

    for job in fragment_jobs:
        job_name = f"{input_path.stem}_ref{job.reference_fragment}"
        tags = [input_path.parent.name, f"ref_frag_{job.reference_fragment}"]
        run_opts = RunOpts(name=job_name[:63], tags=tags)

        frag_keywords = build_frag_keywords(
            cutoff=job.cutoff,
            reference_fragment=job.reference_fragment,
            included_fragments=ligand_fragments,
            trimer_cap=trimer_cutoff_cap,
        )

        output_filename = f"{input_path.stem}_ref{job.reference_fragment}"
        target_path = output_dir / f"{output_filename}.json"
        if target_path.exists():
            continue

        print(f"Process {output_filename}", file=sys.stderr)
        try:
            run = interaction_energy(
                topology_path,
                job.reference_fragment,
                "RestrictedRIMP2",
                "cc-pVDZ",
                "cc-pVDZ-RIFIT",
                scf_keywords=exess.SCFKeywords(
                    max_iters=50,
                    max_diis_history_length=12,
                    convergence_metric="DIIS",
                    convergence_threshold=1e-8,
                    density_threshold=1e-10,
                    density_basis_set_projection_fallback_enabled=True,
                ),
                frag_keywords=frag_keywords,
                run_opts=run_opts,
            )
            submitted.append(run)

            if collect:
                run_output = run.collect()
                run_path = run_output.calc.save()
                if run_path.exists():
                    shutil.move(str(run_path), str(target_path))
                    print(f"  SAVED: {target_path}", file=sys.stderr)
                else:
                    print(
                        f"  Warning: exess output file not found: {run_path}",
                        file=sys.stderr,
                    )
        except RunError as e:
            print(f"  Warning: exess run failed! {e}", file=sys.stderr)

    return submitted if not collect else None


def discover_inputs(
    proj_dir: Path | None, explicit_inputs: list[str] | None
) -> list[Path]:
    """Locate *_fraglig.json inputs."""

    def _is_fraglig(path: Path) -> bool:
        return path.name.endswith("_fraglig.json")

    if explicit_inputs:
        inputs: list[Path] = []
        skipped: list[str] = []
        for raw_path in explicit_inputs:
            candidate = Path(raw_path)
            if _is_fraglig(candidate):
                inputs.append(candidate)
            else:
                skipped.append(raw_path)
        if skipped:
            print(
                "Skipping non-fraglig inputs: " + ", ".join(sorted(skipped)),
                file=sys.stderr,
            )
        if not inputs:
            raise FileNotFoundError(
                "No valid *_fraglig.json files were provided via -i/--input_file."
            )
        return inputs
    if proj_dir is None:
        raise ValueError("--proj-dir is required when no input files are provided.")
    search_root = Path(proj_dir)
    inputs = sorted(search_root.rglob("*_fraglig.json"))
    return inputs
