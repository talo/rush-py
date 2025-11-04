#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from typing import Literal

import cyclopts
from gql.transport.exceptions import TransportQueryError

from .client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    collect_run,
    download_object,
    print_run_trace,
    submit_rex,
    upload_object,
)
from .utils import clean_dict, float_to_str


def prepare_protein(
    trc_path: Path | str,
    ph: float | None = None,
    naming_scheme: Literal["AMBER", "CHARMM"] | None = None,
    capping_style: Literal["never", "truncated", "always"] | None = None,
    truncation_threshold: int | None = None,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run prepare-protein on a TRC and write the prepared TRC file to a json file
    named based on the first 8 characters of the output
    Topology, Residues, and Chains objects joing by an `_`.
    """

    # Upload inputs
    with open(trc_path) as f:
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
        topology_vobj = upload_object(PROJECT_ID, t_f.name)
        residues_vobj = upload_object(PROJECT_ID, r_f.name)
        chains_vobj = upload_object(PROJECT_ID, c_f.name)

    # Run rex
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  exess = λ topology residues chains →
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
  exess "$topology_vobj_path" "$residues_vobj_path" "$chains_vobj_path"
""").substitute(
        run_spec=run_spec.to_rex(),
        ph=float_to_str(ph) if ph is not None else None,
        naming_scheme=naming_scheme,
        capping_style=capping_style,
        truncation_threshold=truncation_threshold,
        topology_vobj_path=topology_vobj["path"],
        residues_vobj_path=residues_vobj["path"],
        chains_vobj_path=chains_vobj["path"],
    )
    try:
        run_id = submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            run = collect_run(run_id)
            if "Ok" in run["result"]:
                trc_o_tuple = run["result"]["Ok"][0]
                t_o_dict = json.loads(download_object(trc_o_tuple[0]["path"]).decode())
                r_o_dict = json.loads(download_object(trc_o_tuple[1]["path"]).decode())
                c_o_dict = json.loads(download_object(trc_o_tuple[2]["path"]).decode())
                trc_o_dict = {
                    "topology": t_o_dict,
                    "residues": r_o_dict,
                    "chains": c_o_dict,
                }
                out_path = (
                    f"{trc_o_tuple[0]['path'][:8]}_"
                    f"{trc_o_tuple[1]['path'][:8]}_"
                    f"{trc_o_tuple[2]['path'][:8]}.json"
                )
                with open(out_path, "w") as f:
                    json.dump(clean_dict(trc_o_dict), f, indent=2)
                return out_path
            elif "Err" in run["result"]:
                print(f"Error: {run['result']['Err']}", file=sys.stderr)
            elif run["status"] == "error":
                print_run_trace(run)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def run_prepare_protein():
    cyclopts.run(prepare_protein)
