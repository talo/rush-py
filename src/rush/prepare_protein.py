#!/usr/bin/env python3
"""
Protein preparation module for the Rush Python client.

This module supports system preparation workflows such as converting PDB inputs
to TRC, protonating and optimizing hydrogen positions, and augmenting
structures with connectivity and formal charge information before downstream
calculations.
"""

import json
import sys
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from typing import Any, Literal, overload

from gql.transport.exceptions import TransportQueryError

from ._output_types import TRCSavedResult
from .client import (
    RunID,
    RunOpts,
    RunSpec,
    _get_project_id,
    _submit_rex,
    collect_run,
    fetch_object,
    save_object,
    upload_object,
)
from .convert import _single_trc, from_json, from_pdb
from .mol import TRC
from .utils import optional_str


def _upload_trc(
    trc: TRC,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    t_f = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    r_f = NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    c_f = NamedTemporaryFile(mode="w", suffix=".json", delete=False)

    json.dump(trc.topology.to_json(), t_f)
    json.dump(trc.residues.to_json(), r_f)
    json.dump(trc.chains.to_json(), c_f)

    # Important: Close temp files before uploading. Windows locks open files,
    # causing PermissionError if upload_object() tries to access them while open.
    t_f.close()
    r_f.close()
    c_f.close()

    return (
        upload_object(t_f.name),
        upload_object(r_f.name),
        upload_object(c_f.name),
    )


@overload
def prepare_protein(
    input_path: Path | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    opt: bool | None = None,
    debump: bool | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: Literal[False] = False,
) -> RunID: ...
@overload
def prepare_protein(
    input_path: Path | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    opt: bool | None = None,
    debump: bool | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: Literal[True] = True,
) -> tuple[dict[str, Any], ...] | list[tuple[dict[str, Any], ...]]: ...
@overload
def prepare_protein(
    input_path: Path | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    opt: bool | None = None,
    debump: bool | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
) -> tuple[dict[str, Any], ...] | list[tuple[dict[str, Any], ...]] | RunID: ...


def prepare_protein(
    input_path: Path | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    opt: bool | None = None,
    debump: bool | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
) -> tuple[dict[str, Any], ...] | list[tuple[dict[str, Any], ...]] | RunID:
    """
    Run prepare-protein on a PDB or TRC file and return the separate T, R, and C files.
    """

    # Upload inputs
    if isinstance(input_path, str):
        input_path = Path(input_path)
    with open(input_path) as f:
        if input_path.suffix == ".pdb":
            trc = from_pdb(f.read())
        else:
            trc = from_json(json.load(f))
    trc = _single_trc(trc, input_path)
    topology_vobj, residues_vobj, chains_vobj = _upload_trc(trc)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  prepare_protein = λ topology residues chains →
    prepare_protein_rex_s
      ($run_spec)
      (prepare_protein_rex::PrepareProteinOptions {
        ph = $ph,
        naming_scheme = $naming_scheme,
        capping_style = $capping_style,
        truncation_threshold = $truncation_threshold,
        opt = $opt,
        debump = $debump,
      })
      [( (obj_j topology), (obj_j residues), (obj_j chains) )]
in
  prepare_protein "$topology_vobj_path" "$residues_vobj_path" "$chains_vobj_path"
""").substitute(
        run_spec=run_spec._to_rex(),
        ph=optional_str(ph),
        naming_scheme=optional_str(
            naming_scheme.title() if naming_scheme is not None else None,
            prefix="prepare_protein_rex::NamingScheme::",
        ),
        capping_style=optional_str(
            capping_style.title() if capping_style is not None else None,
            prefix="prepare_protein_rex::CappingStyle::",
        ),
        truncation_threshold=optional_str(truncation_threshold),
        opt=optional_str(opt),
        debump=optional_str(debump),
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"],
        chains_vobj_path=chains_vobj["path"],
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if not collect:
            return run_id

        out = collect_run(run_id)
        out = [tuple(out_i) for out_i in out]
        assert isinstance(out, list)
        if len(out) == 1:
            out = out[0]
        return out

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise e


@overload
def _unwrap_outputs(
    res: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: ...
@overload
def _unwrap_outputs(
    res: list[tuple[dict[str, Any], ...]],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]: ...


def _unwrap_outputs(res):
    if isinstance(res, tuple) and len(res) == 3:
        return (res[0], res[1], res[2])
    else:
        out = []
        for res_i in res:
            if isinstance(res_i, tuple) and len(res) == 3:
                out.append((res[0], res[1], res[2]))
            else:
                raise ValueError(
                    f"Error: prepare_protein output helper received unexpected format: {type(res)}"
                )
        return out

    raise ValueError(
        f"Error: prepare_protein output helper received unexpected format: {type(res)}"
    )


def fetch_outputs(
    res: tuple[dict[str, Any], ...] | list[tuple[dict[str, Any], ...]],
) -> TRC | list[TRC]:
    """
    Fetch prepare-protein outputs into an in-memory TRC.

    Args:
        res: Collected output from prepare_protein(), containing topology,
            residues, and chains objects.

    Returns:
        Parsed TRC data.
    """

    def fetch_output(res: tuple[dict[str, Any], ...]) -> TRC:
        outputs = _unwrap_outputs(res)
        topology_obj, residues_obj, chains_obj = outputs
        return from_json(
            {
                "topology": json.loads(fetch_object(topology_obj["path"])),
                "residues": json.loads(fetch_object(residues_obj["path"])),
                "chains": json.loads(fetch_object(chains_obj["path"])),
            }
        )

    if isinstance(res, tuple):
        return fetch_output(res)
    else:
        return [fetch_output(res_i) for res_i in res]


def save_outputs(
    res: tuple[dict[str, Any], ...] | list[tuple[dict[str, Any], ...]],
) -> TRCSavedResult | list[TRCSavedResult]:
    """
    Download output files from a prepare-protein run.

    The prepare-protein computation returns three VirtualObject dicts for the
    topology, residues, and chains files. This function downloads each file
    and returns paths that can be passed to `from_json()`.

    Args:
        res: Collected output from prepare_protein(), containing topology,
            residues, and chains objects.

    Returns:
        Local paths to the saved topology, residues, and chains files.
    """

    def save_output(res: tuple[dict[str, Any], ...]):
        outputs = _unwrap_outputs(res)
        topology_obj, residues_obj, chains_obj = outputs
        return TRCSavedResult(
            topology=save_object(topology_obj["path"]),
            residues=save_object(residues_obj["path"]),
            chains=save_object(chains_obj["path"]),
        )

    if isinstance(res, tuple):
        return save_output(res)
    else:
        return [save_output(res_i) for res_i in res]
