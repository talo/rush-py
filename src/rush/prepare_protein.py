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
import tempfile
from typing import Literal

from gql.transport.exceptions import TransportQueryError

from .client import (
    RunError,
    RunOpts,
    RunSpec,
    _get_project_id,
    _submit_rex,
    collect_run,
    save_object,
    upload_object,
)
from .convert import from_json, from_pdb
from .utils import optional_str


def prepare_protein(
    input_path: Path | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    debump: bool | None = None,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
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
            if isinstance(trc, list):
                if len(trc) != 1:
                    raise ValueError(
                        f"Expected 1 TRC in {input_path}, found {len(trc)}"
                    )
                trc = trc[0]
    t_f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    r_f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    c_f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

    json.dump(trc.topology.to_json(), t_f)
    json.dump(trc.residues.to_json(), r_f)
    json.dump(trc.chains.to_json(), c_f)

    # Important: Close temp files before uploading. Windows locks open files,
    # causing PermissionError if upload_object() tries to access them while open.
    t_f.close()
    r_f.close()
    c_f.close()

    topology_vobj = upload_object(t_f.name)
    residues_vobj = upload_object(r_f.name)
    chains_vobj = upload_object(c_f.name)

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
        debump=optional_str(debump),
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"],
        chains_vobj_path=chains_vobj["path"],
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def save_outputs(
    res: dict | list | tuple | str | RunError,
) -> tuple[Path, Path, Path] | str | RunError:
    """
    Download output files from a prepare-protein run.

    The prepare-protein rex computation returns a list/tuple of 3 VirtualObject
    dicts (topology, residues, chains files). This function downloads each
    file and returns Path objects that can be used with from_json().

    If collect=False was used, the input will be a run ID string, which is
    returned as-is for later collection by the caller.

    Args:
        res: Either:
             - A run ID string (if collect=False was used)
             - A list/tuple of 3 VirtualObject dicts from collect_run()
             - A RunError
             Each VirtualObject dict has keys: 'path', 'size', 'format'.

    Returns:
        Either:
        - A run ID string (if input was a run ID)
        - Tuple of 3 downloaded file Paths (if input was VirtualObject list)
        - RunError if input is an error
    """
    # Handle error case
    if isinstance(res, RunError):
        return res

    # Handle run ID string (collect=False case)
    if isinstance(res, str):
        return res

    # Handle list/tuple of VirtualObject dicts from collect_run()
    if isinstance(res, (list, tuple)) and len(res) >= 3:
        return (
            save_object(res[0]["path"]),
            save_object(res[1]["path"]),
            save_object(res[2]["path"]),
        )

    # Fallback: return as-is (for debugging or unexpected formats)
    print(f"Warning: save_outputs received unexpected format: {type(res)}")
    print(res)
    return res
