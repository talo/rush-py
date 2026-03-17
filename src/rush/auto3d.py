import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Iterator

from gql.transport.exceptions import TransportQueryError

from rush import TRC, from_json
from rush.client import (
    RunError,
    RunOpts,
    RunSpec,
    _get_project_id,
    _json_content_name,
    _submit_rex,
    collect_run,
    fetch_object,
    save_json,
    save_object,
)
from rush.utils import bool_to_str, float_to_str


@dataclass
class Auto3DStats:
    f_max: float
    converged: bool
    e_rel_kcal_mol: float
    e_tot_hartrees: float


@dataclass
class Auto3DResult:
    conformer: TRC
    stats: Auto3DStats


@dataclass
class Auto3DSavedResult:
    conformer: tuple[Path, Path, Path]
    stats: Path


def auto3d(
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
    collect=False,
):
    """
    Runs Auto3D on a list of SMILES strings, returning either the TRC structure
    or an error string.
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
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if not collect:
            return run_id

        result = collect_run(run_id)
        if isinstance(result, RunError):
            return result

        def is_result_type(result):
            return (
                isinstance(result, dict)
                and len(result) == 1
                and ("Ok" in result or "Err" in result)
            )

        # TODO: no special cases for Result unwrapping

        return [
            next(iter(r_i.values())) if is_result_type(r_i) else r_i for r_i in result
        ]

    except TransportQueryError as e:
        if e.errors:
            print("Error:", file=sys.stderr)
            for error in e.errors:
                print(f"  {error['message']}", file=sys.stderr)


def _map_outputs(
    res,
    *,
    on_success,
):
    if isinstance(res, RunError):
        return res

    if isinstance(res, str):
        return res

    if isinstance(res, list):
        return [
            RunError(res_i) if isinstance(res_i, str) else on_success(res_i)
            for res_i in res
        ]

    return RunError(
        f"Error: auto3d output helper received unexpected format: {type(res)}"
    )


def fetch_outputs(res) -> list[Iterator[Auto3DResult] | RunError] | str | RunError:
    """
    Download output files from an auto3d run.

    The auto3d rex computation returns a Rush object store pointers for TRCs
    and stats for each conformer generated. There are up to k conformers
    per input. Each input can either succeed, in which case a Iterator[Auto3DResult]
    is returned that downloads and packages each conformer on the fly, or fail,
    in which case the run error is returned.

    If collect=False was used, the input will be a run ID string, which is
    returned as-is for later collection by the caller.

    Args:
        res: Either:
             - A run ID string (if collect=False was used)
             - The successful output from auto3d()
             - A RunError
             Each VirtualObject dict has keys: 'path', 'size', 'format'.

    Returns:
        Either:
        - A run ID string (if input was a run ID)
        - list[Iterator[Auto3DResult] | RunError], if the run succeeded
        - RunError if input is an error
    """

    def to_auto3dresult(res_i) -> Iterator[Auto3DResult]:
        for trc_obj, stats in res_i:
            trc_dict = {
                "topology": json.loads(fetch_object(trc_obj[0]["path"])),
                "residues": json.loads(fetch_object(trc_obj[1]["path"])),
                "chains": json.loads(fetch_object(trc_obj[2]["path"])),
            }
            yield Auto3DResult(
                from_json(trc_dict),
                Auto3DStats(
                    stats["f_max"],
                    stats["converged"],
                    stats["e_rel_kcal_mol"],
                    stats["e_tot_hartrees"],
                ),
            )

    return _map_outputs(res, on_success=to_auto3dresult)


def save_outputs(
    res,
) -> list[Iterator[Auto3DSavedResult] | RunError] | str | RunError:
    """
    Save Auto3D outputs into the workspace.

    Each successful input yields an iterator of conformers. Every conformer is
    saved as three TRC component files `(topology, residues, chains)` plus a
    JSON file containing the associated Auto3DStats.
    """

    def save_auto3dresult(res_i) -> Iterator[Auto3DSavedResult]:
        for trc_obj, stats in res_i:
            yield Auto3DSavedResult(
                conformer=(
                    save_object(trc_obj[0]["path"]),
                    save_object(trc_obj[1]["path"]),
                    save_object(trc_obj[2]["path"]),
                ),
                stats=save_json(
                    stats,
                    name=_json_content_name("auto3d_stats", stats),
                ),
            )

    return _map_outputs(res, on_success=save_auto3dresult)
