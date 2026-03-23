"""
Auto3D module for the Rush Python client.

Auto3D generates 3D conformers from SMILES strings using the AIMNET
optimizing engine.  It supports configurable conformer counts, convergence
thresholds, and isomer/tautomer enumeration.

Usage::

    from rush import auto3d

    run = auto3d.generate(["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"], k=5)
    ref = run.collect()
    results = ref.fetch()
"""

import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Callable, NewType, TypeGuard, TypeVar

from gql.transport.exceptions import TransportQueryError

from rush import TRC, from_json

from ._trc import TRCPaths
from ._utils import bool_to_str, float_to_str
from .client import (
    RunOpts,
    RunSpec,
    _get_project_id,
    _json_content_name,
    _submit_rex,
    fetch_object,
    save_json,
    save_object,
)
from .run import RushRun


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    f_max: float
    converged: bool
    e_rel_kcal_mol: float
    e_tot_hartrees: float


@dataclass
class Result:
    conformer: TRC
    stats: Stats


@dataclass(frozen=True)
class ResultPaths:
    conformer: TRCPaths
    stats: Path


Error = NewType("Error", str)


T = TypeVar("T")


def _is_result_type(result: Any) -> TypeGuard[dict[str, Any]]:
    return (
        isinstance(result, dict)
        and len(result) == 1
        and ("Ok" in result or "Err" in result)
    )


def _unwrap_raw(raw: list[Any]) -> list[Any]:
    """Unwrap the Ok/Err layer that Auto3D wraps per-input results in."""
    out = [
        next(iter(out_i.values())) if _is_result_type(out_i) else out_i for out_i in raw
    ]
    if len(out) == 1:
        out = out[0]
    return out


def _map_outputs(
    res: list[Any],
    *,
    on_success: Callable[[Any], T],
) -> list[T | Error]:
    return [
        # Handle per-conformer error strings
        Error(res_i) if isinstance(res_i, str) else on_success(res_i)
        for res_i in res
    ]


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to Auto3D outputs in the Rush object store.

    Call :meth:`fetch` to download and parse into Python dataclasses, or
    :meth:`save` to download to local files.
    """

    raw: list[Any]

    @classmethod
    def from_raw_output(cls, raw: Any) -> "ResultRef":
        """Parse raw ``collect_run`` output into a ``ResultRef``.

        The raw output from ``collect_run`` is a ``list[Any]`` where each
        element is EITHER a dict containing a ``smiles`` field (error) OR a
        list of ``(trc_objs, stats)`` tuples (conformers).  We unwrap the
        ``Ok``/``Err`` layer and store the result.
        """
        if not isinstance(raw, list):
            raise ValueError(f"auto3d should return a list, got {type(raw).__name__}.")
        unwrapped = _unwrap_raw(raw)
        return cls(raw=unwrapped)

    def fetch(self) -> list[Iterator[Result] | Error]:
        """Download output files and parse into :class:`Result` objects.

        Each input SMILES either succeeds (returning an iterator of conformer
        :class:`Result` objects) or fails (returning an :class:`Error`).

        Returns:
            One item per input: either an iterator over fetched conformers or
            an Error for that input.
        """

        def fetch_output(res_i: Any) -> Iterator[Result]:
            for trc_obj, stats in res_i:
                trc_dict = {
                    "topology": json.loads(fetch_object(trc_obj[0]["path"])),
                    "residues": json.loads(fetch_object(trc_obj[1]["path"])),
                    "chains": json.loads(fetch_object(trc_obj[2]["path"])),
                }
                yield Result(
                    from_json(trc_dict),
                    Stats(
                        stats["f_max"],
                        stats["converged"],
                        stats["e_rel_kcal_mol"],
                        stats["e_tot_hartrees"],
                    ),
                )

        return _map_outputs(self.raw, on_success=fetch_output)

    def save(self) -> list[Iterator[ResultPaths] | Error]:
        """Save Auto3D outputs into the workspace.

        Each successful input yields an iterator of conformers.  Every
        conformer is saved as three TRC component files ``(topology,
        residues, chains)`` plus a JSON file containing the associated stats.

        Returns:
            One item per input: either an iterator over saved conformers or
            an Error for that input.
        """

        def save_output(res_i: Any) -> Iterator[ResultPaths]:
            for trc_obj, stats in res_i:
                yield ResultPaths(
                    conformer=TRCPaths(
                        topology=save_object(trc_obj[0]["path"]),
                        residues=save_object(trc_obj[1]["path"]),
                        chains=save_object(trc_obj[2]["path"]),
                    ),
                    stats=save_json(
                        stats,
                        name=_json_content_name("auto3d_stats", stats),
                    ),
                )

        return _map_outputs(self.raw, on_success=save_output)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def generate(
    smis: list[str],
    k: int = 1,
    batchsize_atoms: int = 1024,
    capacity: int = 40,
    convergence_threshold: float = 0.003,
    enumerate_isomer: bool = True,
    enumerate_tautomer: bool = False,
    max_confs: int | None = None,
    opt_steps: int = 5000,
    patience: int = 1000,
    threshold: float = 0.3,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
) -> RushRun[ResultRef]:
    """
    Submit an Auto3D conformer generation job for a list of SMILES strings.

    Returns a :class:`~rush.run.RushRun` handle.  Call ``.collect()`` to wait
    for the result ref, then ``.fetch()`` or ``.save()`` to retrieve outputs.
    """
    rex = Template("""let
  auto3d = λ smis →
    try_auto3d_rex
      default_runspec_gpu
      (auto3d_rex::Auto3dOptions {
        k = Some (int $k),
        batchsize_atoms = Some $batchsize_atoms,
        capacity = Some $capacity,
        convergence_threshold = Some $convergence_threshold,
        enumerate_isomer = Some $enumerate_isomer,
        enumerate_tautomer = Some $enumerate_tautomer,
        job_name = None,
        max_confs = $max_confs,
        memory = None,
        mpi_np = Some 4,
        opt_steps = Some $opt_steps,
        optimizing_engine = Some auto3d_rex::Auto3dOptimizingEngines::AIMNET,
        patience = Some $patience,
        threshold = Some $threshold,
        verbose = Some false,
        window = None,
      })
      $smis
in
  auto3d $smis
""").substitute(
        smis=f"[{', '.join([f'"{smi}"' for smi in smis])}]",
        k=k,
        batchsize_atoms=batchsize_atoms,
        capacity=capacity,
        convergence_threshold=float_to_str(convergence_threshold),
        enumerate_isomer=bool_to_str(enumerate_isomer),
        enumerate_tautomer=bool_to_str(enumerate_tautomer),
        max_confs=max_confs,
        opt_steps=opt_steps,
        patience=patience,
        threshold=float_to_str(threshold),
        run_spec=run_spec._to_rex(),
    )
    try:
        return RushRun(
            _submit_rex(_get_project_id(), rex, run_opts),
            ResultRef,
        )

    except TransportQueryError as e:
        if e.errors:
            print("Error:", file=sys.stderr)
            for error in e.errors:
                print(f"  {error['message']}", file=sys.stderr)
        raise
