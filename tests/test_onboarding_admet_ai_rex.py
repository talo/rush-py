import sys
from pathlib import Path

import pytest

from rush import admet_ai_rex
from rush.client import (
    GRAPHQL_ENDPOINT,
    MODULE_LOCK,
    RunError,
    RunOpts,
    RunSpec,
    _get_env,
    collect_run,
    set_opts,
)


MODULE_KEYS = [
    "talo_admet_ai_rex",
    "talo_admet_ai_plot_drugbank_rex",
    "talo_admet_ai_plot_radial_rex",
    "talo_admet_ai_web_rex",
]


def test_admet_ai_rex_imports():
    assert hasattr(admet_ai_rex, "talo_admet_ai_rex")
    assert hasattr(admet_ai_rex, "talo_admet_ai_plot_drugbank_rex")
    assert hasattr(admet_ai_rex, "talo_admet_ai_plot_radial_rex")
    assert hasattr(admet_ai_rex, "talo_admet_ai_web_rex")


def test_admet_ai_rex_module_lock():
    if "staging" in GRAPHQL_ENDPOINT:
        for key in MODULE_KEYS:
            assert key in MODULE_LOCK
    else:
        pytest.xfail("Prod endpoint in use and update_prod=False.")


def _assert_run_ok(result):
    if isinstance(result, RunError):
        pytest.fail(f"RunError: {result.message}")
    if isinstance(result, dict) and "Err" in result:
        pytest.fail(f"Run returned Err: {result['Err']}")
    if isinstance(result, (list, tuple)):
        for item in result:
            if isinstance(item, dict) and "Err" in item:
                pytest.fail(f"Run returned Err: {item['Err']}")


def test_admet_ai_plot_radial_run():
    if not _get_env("RUSH_TOKEN") or not _get_env("RUSH_PROJECT"):
        pytest.skip("RUSH_TOKEN and RUSH_PROJECT must be set for integration run.")

    set_opts(workspace_dir=Path.cwd() / "test-runs")
    input_path = Path(__file__).resolve().parent / "data" / "admet_ai_rex" / "preds.json"
    config_rex = """(talo_admet_ai_plot_radial_rex::WrapperConfig {
  percentile_suffix = Some "drugbank_approved_percentile",
  atc_code = None,
  image_type = "svg"
})"""
    run_id = admet_ai_rex.talo_admet_ai_plot_radial_rex(
        input_path,
        config_rex,
        run_spec=RunSpec(),
        run_opts=RunOpts(
            name="Rush-Py Test ADMET AI Plot Radial",
            tags=["rush-py", "test", "admet-ai-rex"],
        ),
        collect=False,
    )
    print(f"run_id: {run_id}", file=sys.stderr)
    result = collect_run(run_id)
    print(result, file=sys.stderr)
    _assert_run_ok(result)
