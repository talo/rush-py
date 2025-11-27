"""
Submit EXESS interaction-energy jobs for ligand fragments (FRIED workflow).

Primary entrypoint:
- fragmented_exess(input_file, distance_threshold=4.0, trimer_cutoff_cap=15.0, collect=True, output_dir=None) -> None
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .. import exess as rushpy2_exess
from ..client import RunOpts
from ..exess import FragKeywords as ExessFragKeywords
from ..exess import SCFKeywords as ExessSCFKeywords
from ..exess import System as ExessSystem

DEFAULT_METHOD = "RestrictedRIMP2"
DEFAULT_BASIS = "cc-pVDZ"
DEFAULT_AUX_BASIS = "cc-pVDZ-RIFIT"

DEFAULT_SYSTEM_PARAMS: dict[str, Any] = {}

DEFAULT_SCF_PARAMS = {
    "max_iters": 50,
    "max_diis_history_length": 12,
    "convergence_metric": "DIIS",
    "convergence_threshold": 1e-8,
    "density_threshold": 1e-10,
    "density_basis_set_projection_fallback_enabled": True,
}

__all__ = [
    "fragmented_exess",
    "determine_ligand_atoms",
    "collect_ligand_fragments",
    "compute_fragment_cutoffs",
    "build_frag_keywords",
    "run_exess",
    "discover_inputs",
]


@dataclass(frozen=True)
class FragmentJob:
    reference_fragment: int
    cutoff: int


def _ensure_rushpy2_available() -> None:
    if rushpy2_exess is None:
        raise ModuleNotFoundError(
            "rush_py2 is required to submit EXESS jobs. Install the 'rush-py2' package."
        )


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


def collect_ligand_fragments(conf: dict[str, Any], ligand_atoms: set[int]) -> list[int]:
    ligand_fragments: list[int] = []
    for frag_idx, fragment in enumerate(conf["topology"]["fragments"]):
        if set(fragment) & ligand_atoms:
            ligand_fragments.append(frag_idx)
    if not ligand_fragments:
        raise ValueError("Failed to match ligand atoms to fragment indices.")
    return ligand_fragments


def compute_fragment_cutoffs(
    conf: dict[str, Any],
    ligand_atoms: set[int],
    ligand_fragments: set[int],
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


def _build_system_config() -> Any:
    if ExessSystem is None or not DEFAULT_SYSTEM_PARAMS:
        return None
    return ExessSystem(**DEFAULT_SYSTEM_PARAMS)


def _build_scf_keywords() -> Any:
    if ExessSCFKeywords is None:
        return dict(DEFAULT_SCF_PARAMS)
    return ExessSCFKeywords(**DEFAULT_SCF_PARAMS)


def build_frag_keywords(
    cutoff: int,
    reference_fragment: int,
    included_fragments: list[int],
    trimer_cap: float,
) -> Any:
    trimer_cutoff = float(min(cutoff, trimer_cap))
    params = {
        "level": "Trimer",
        "dimer_cutoff": float(cutoff),
        "trimer_cutoff": trimer_cutoff,
        "tetramer_cutoff": trimer_cutoff,
        "cutoff_type": "Centroid",
        "included_fragments": sorted(set([reference_fragment] + included_fragments)),
    }
    if ExessFragKeywords is None:
        return params
    return ExessFragKeywords(**params)


def run_exess(
    topology_path: Path,
    output_dir: Path,
    collect: bool,
    output_filename: str | None = None,
    run_opts: Any | None = None,
    exess_kwargs: dict[str, Any] | None = None,
    reference_fragment: int | None = None,
) -> None:
    """
    Run exess on a single topology and save results.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_path = output_dir / f"{output_filename or topology_path.stem}.json"
    if target_path.exists():
        print(f"  SKIP: Output file already exists: {target_path}", file=sys.stderr)
        return

    call_kwargs: dict[str, Any] = {"collect": collect}
    if run_opts is not None:
        call_kwargs["run_opts"] = run_opts
    if exess_kwargs:
        call_kwargs.update(exess_kwargs)

    call_kwargs["reference_fragment"] = reference_fragment

    _ensure_rushpy2_available()
    submit_fn = rushpy2_exess.interaction_energy
    topology_path_str = str(topology_path)

    if not topology_path.exists():
        raise RuntimeError(f"Topology file no longer exists: {topology_path_str}")

    job_display_name = output_filename or topology_path.stem
    print(f"Process {job_display_name}", file=sys.stderr)

    run_output = submit_fn(topology_path_str, **call_kwargs)
    if not run_output:
        print("Warning: exess.energy returned no filename", file=sys.stderr)
        return

    run_path = Path(run_output)
    if run_path.exists():
        shutil.move(str(run_path), str(target_path))
        print(f"  SAVED: {target_path}", file=sys.stderr)
    elif collect:
        print(f"Warning: exess output file not found: {run_path}", file=sys.stderr)


def fragmented_exess(
    input_file: str | Path,
    distance_threshold: float = 4.0,
    trimer_cutoff_cap: float = 15.0,
    collect: bool = True,
    output_dir: Path | None = None,
) -> None:
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

    for job in fragment_jobs:
        job_name = f"{input_path.stem}_ref{job.reference_fragment}"
        run_opts = None
        if RunOpts is not None:
            tags = [input_path.parent.name, f"ref_frag_{job.reference_fragment}"]
            run_opts = RunOpts(name=job_name[:63], tags=tags)

        frag_keywords = build_frag_keywords(
            cutoff=job.cutoff,
            reference_fragment=job.reference_fragment,
            included_fragments=ligand_fragments,
            trimer_cap=trimer_cutoff_cap,
        )

        exess_kwargs = {
            "method": DEFAULT_METHOD,
            "basis": DEFAULT_BASIS,
            "aux_basis": DEFAULT_AUX_BASIS,
            "system": _build_system_config(),
            "scf_keywords": _build_scf_keywords(),
            "frag_keywords": frag_keywords,
        }

        output_filename = f"{input_path.stem}_ref{job.reference_fragment}"
        if (output_dir / f"{output_filename}.json").exists():
            continue

        run_exess(
            topology_path=topology_path,
            output_dir=output_dir,
            collect=collect,
            output_filename=output_filename,
            run_opts=run_opts,
            exess_kwargs=exess_kwargs,
            reference_fragment=job.reference_fragment,
        )


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
