import importlib
import os
from pathlib import Path

import pytest

MODULE_KEYS = [
    "grimme_lab_xtb_single_point_rex",
    "grimme_lab_xtb_gradient_rex",
    "grimme_lab_xtb_optimize_rex",
    "grimme_lab_xtb_hessian_rex",
    "grimme_lab_xtb_optimized_hessian_rex",
    "grimme_lab_xtb_biased_hessian_rex",
    "grimme_lab_xtb_md_rex",
    "grimme_lab_xtb_metadyn_rex",
    "grimme_lab_xtb_optimized_md_rex",
    "grimme_lab_xtb_metaopt_rex",
    "grimme_lab_xtb_path_rex",
    "grimme_lab_xtb_mode_following_rex",
    "grimme_lab_xtb_reactor_rex",
    "grimme_lab_xtb_dipro_rex",
    "grimme_lab_xtb_vip_rex",
    "grimme_lab_xtb_vea_rex",
    "grimme_lab_xtb_vipea_rex",
    "grimme_lab_xtb_vfukui_rex",
    "grimme_lab_xtb_vomega_rex",
    "grimme_lab_xtb_ceh_rex",
    "grimme_lab_xtb_esp_rex",
    "grimme_lab_xtb_stm_rex",
    "grimme_lab_xtb_raman_rex",
    "grimme_lab_xtb_oniom_rex",
]

FIXTURE_INPUT = (
    Path(__file__).resolve().parent / "fixtures" / "xtb_rex" / "objects" / "input.json"
)


def _load_client(endpoint: str):
    os.environ["RUSH_ENDPOINT"] = endpoint
    import rush.client as client

    return importlib.reload(client)


def test_xtb_rex_onboarding():
    endpoint = os.environ.get("RUSH_ENDPOINT", "")
    if "staging" not in endpoint:
        pytest.skip("RUSH_ENDPOINT must contain 'staging' for xtb_rex onboarding test.")
    if not os.environ.get("RUSH_TOKEN") or not os.environ.get("RUSH_PROJECT"):
        pytest.skip("RUSH_TOKEN and RUSH_PROJECT are required for xtb_rex onboarding test.")

    client = _load_client(endpoint)
    for key in MODULE_KEYS:
        assert key in client.MODULE_LOCK

    import rush.xtb as xtb

    xtb = importlib.reload(xtb)
    for key in MODULE_KEYS:
        assert hasattr(xtb, key)

    config_rex = """(grimme_lab_xtb_single_point_rex::WrapperConfig {
  common = (grimme_lab_xtb_single_point_rex::CommonConfig {
    method = None,
    charge = None,
    uhf = None,
    spin_polarized = None,
    solvation = None
  }),
  use_scc = None
})"""

    result = xtb.grimme_lab_xtb_single_point_rex(
        FIXTURE_INPUT,
        config_rex,
        run_spec=client.RunSpec(target="Bullet3"),
        run_opts=client.RunOpts(),
        collect=True,
    )

    assert not isinstance(result, client.RunError)
