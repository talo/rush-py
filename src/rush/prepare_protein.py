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
from typing import Literal

from gql.transport.exceptions import TransportQueryError

from .client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    _submit_rex,
    collect_run,
    save_object,
    upload_object,
)
from .convert import from_pdb, to_json
from .utils import optional_str


def prepare_protein(
    input_path: Path | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
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
            trc_str = to_json(trc)
            trc_dict = json.loads(trc_str)[0]
        else:
            trc_dict = json.load(f)
    with (
        NamedTemporaryFile(mode="w") as t_f,
        NamedTemporaryFile(mode="w") as r_f,
        NamedTemporaryFile(mode="w") as c_f,
    ):
        json.dump(trc_dict["topology"], t_f)
        json.dump(trc_dict["residues"], r_f)
        json.dump(trc_dict["chains"], c_f)
        t_f.seek(0)
        r_f.seek(0)
        c_f.seek(0)
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
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"],
        chains_vobj_path=chains_vobj["path"],
    )
    try:
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def save_outputs(res):
    return (
        save_object(res[0]["path"]),
        save_object(res[1]["path"]),
        save_object(res[2]["path"]),
    )
