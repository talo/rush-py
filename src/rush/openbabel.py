#!/usr/bin/env python3
"""
Open Babel module helpers for the Rush Python client.
"""

import sys
from pathlib import Path
from string import Template

from gql.transport.exceptions import TransportQueryError

from .client import (
    RunOpts,
    RunSpec,
    _get_project_id,
    _submit_rex,
    collect_run,
    upload_object,
)


def openbabel_openbabel_protonate_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet"),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Protonate a ligand with Open Babel using TRC JSON input.

    Args:
        input_json: Path to the TRC JSON input object.
        config_rex: Rex expression for the config struct, e.g.
            "openbabel_openbabel_protonate_rex::ProtonateConfig { ph = Some 7.4, babel_libdir = None, babel_datadir = None }".
        run_spec: Rush compute resources to request.
        run_opts: Rush run metadata.
        collect: Whether to wait for completion and return outputs.
    """

    input_vobj = upload_object(input_json)

    rex = Template(
        """let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  run = λ input →
    openbabel_openbabel_protonate_rex_s
      ($run_spec)
      ($config_rex)
      (obj_j input)
in
  run "$input_vobj_path"
"""
    ).substitute(
        run_spec=run_spec._to_rex(),
        config_rex=config_rex,
        input_vobj_path=input_vobj["path"],
    )

    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        return run_id
    except TransportQueryError as e:
        if e.errors:
            print("Error:", file=sys.stderr)
            for error in e.errors:
                print(f"  {error['message']}", file=sys.stderr)
