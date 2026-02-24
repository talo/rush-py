#!/usr/bin/env python3
"""
Raw-config Rex wrappers for the ADMET AI Tengu module repo.
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


def _upload_json(input_json: Path | str) -> str:
    if isinstance(input_json, str):
        input_json = Path(input_json)
    obj = upload_object(input_json)
    return obj["path"]


def talo_admet_ai_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run talo_admet_ai_rex with a raw config Rex expression and JSON input.
    """
    input_path = _upload_json(input_json)
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  cfg = $config_rex,
  input = obj_j "$input_path",
  result = talo_admet_ai_rex_s
    ($run_spec)
    cfg
    input
in
  result
""").substitute(
        run_spec=run_spec._to_rex(),
        config_rex=config_rex,
        input_path=input_path,
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        return run_id
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def talo_admet_ai_plot_drugbank_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run talo_admet_ai_plot_drugbank_rex with a raw config Rex expression and JSON input.
    """
    input_path = _upload_json(input_json)
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  cfg = $config_rex,
  preds = obj_j "$input_path",
  result = talo_admet_ai_plot_drugbank_rex_s
    ($run_spec)
    cfg
    preds
in
  result
""").substitute(
        run_spec=run_spec._to_rex(),
        config_rex=config_rex,
        input_path=input_path,
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        return run_id
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def talo_admet_ai_plot_radial_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run talo_admet_ai_plot_radial_rex with a raw config Rex expression and JSON input.
    """
    input_path = _upload_json(input_json)
    rex = Template("""let
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 },
  cfg = $config_rex,
  preds = obj_j "$input_path",
  result = talo_admet_ai_plot_radial_rex_s
    ($run_spec)
    cfg
    preds
in
  result
""").substitute(
        run_spec=run_spec._to_rex(),
        config_rex=config_rex,
        input_path=input_path,
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        return run_id
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)


def talo_admet_ai_web_rex(
    input_json: Path | str | None,
    config_rex: str,
    run_spec: RunSpec = RunSpec(),
    run_opts: RunOpts = RunOpts(),
    collect=False,
):
    """
    Run talo_admet_ai_web_rex with a raw config Rex expression.

    The web entrypoint does not accept an input object; input_json is ignored.
    """
    rex = Template("""let
  cfg = $config_rex,
  result = talo_admet_ai_web_rex_s
    ($run_spec)
    cfg
in
  result
""").substitute(
        run_spec=run_spec._to_rex(),
        config_rex=config_rex,
    )
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        return run_id
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
