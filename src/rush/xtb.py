#!/usr/bin/env python3
"""
Grimme lab xTB module helpers for the Rush Python client.

These wrappers accept raw Rex config expressions to avoid guessing module-specific
config structures. Use the module repo's test.rex files as references for config
shapes and defaults.
"""

import sys
from pathlib import Path

from gql.transport.exceptions import TransportQueryError

from .client import (
    RunOpts,
    RunSpec,
    _get_project_id,
    _submit_rex,
    collect_run,
    upload_object,
)


def _run_xtb_module(
    module_key: str,
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec,
    run_opts: RunOpts,
    collect: bool,
):
    input_vobj = upload_object(input_json)
    rex = f"""let
  obj_j = \u03bb j \u2192
    VirtualObject {{ path = j, format = ObjectFormat::json, size = 0 }},
  input = obj_j \"{input_vobj['path']}\",
  cfg = {config_rex},
  result = {module_key}_s
    ({run_spec._to_rex()})
    cfg
    input
in
  result
"""
    try:
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if collect:
            return collect_run(run_id)
        return run_id
    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise


def grimme_lab_xtb_single_point_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_single_point_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_single_point_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_single_point_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_gradient_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_gradient_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_gradient_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_gradient_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_optimize_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_optimize_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_optimize_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_optimize_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_hessian_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_hessian_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_hessian_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_hessian_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_optimized_hessian_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_optimized_hessian_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_optimized_hessian_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_optimized_hessian_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_biased_hessian_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_biased_hessian_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_biased_hessian_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_biased_hessian_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_md_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_md_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_md_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_md_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_metadyn_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_metadyn_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_metadyn_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_metadyn_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_optimized_md_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_optimized_md_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_optimized_md_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_optimized_md_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_metaopt_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_metaopt_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_metaopt_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_metaopt_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_path_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_path_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_path_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_path_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_mode_following_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_mode_following_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_mode_following_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_mode_following_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_reactor_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_reactor_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_reactor_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_reactor_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_dipro_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_dipro_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_dipro_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_dipro_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_vip_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_vip_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_vip_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_vip_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_vea_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_vea_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_vea_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_vea_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_vipea_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_vipea_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_vipea_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_vipea_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_vfukui_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_vfukui_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_vfukui_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_vfukui_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_vomega_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_vomega_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_vomega_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_vomega_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_ceh_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_ceh_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_ceh_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_ceh_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_esp_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_esp_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_esp_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_esp_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_stm_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_stm_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_stm_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_stm_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_raman_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_raman_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_raman_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_raman_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )


def grimme_lab_xtb_oniom_rex(
    input_json: Path | str,
    config_rex: str,
    run_spec: RunSpec = RunSpec(target="Bullet3"),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
):
    """
    Run the grimme_lab_xtb_oniom_rex module.

    config_rex should be a Rex expression like:
      (grimme_lab_xtb_oniom_rex::WrapperConfig { ... })
    """
    return _run_xtb_module(
        "grimme_lab_xtb_oniom_rex",
        input_json,
        config_rex,
        run_spec,
        run_opts,
        collect,
    )
