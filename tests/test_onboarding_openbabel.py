import os
import time
from pathlib import Path

import pytest
from gql.transport.exceptions import TransportConnectionFailed


def test_openbabel_wrapper_import():
    import rush.openbabel as openbabel

    assert hasattr(openbabel, "openbabel_openbabel_protonate_rex")


def test_openbabel_module_lock():
    from rush import client

    assert "openbabel_openbabel_protonate_rex" in client.MODULE_LOCK


@pytest.mark.skipif(
    not os.getenv("RUSH_TOKEN") or not os.getenv("RUSH_PROJECT"),
    reason="RUSH_TOKEN and RUSH_PROJECT are required for live runs",
)
def test_openbabel_protonate_run():
    from rush import openbabel
    from rush.client import RunError, RunSpec

    input_json = Path(__file__).resolve().parent / "data" / "openbabel" / "input.json"
    config_rex = (
        "openbabel_openbabel_protonate_rex::ProtonateConfig "
        "{ ph = Some 7.4, babel_libdir = None, babel_datadir = None }"
    )

    result = None
    for attempt in range(3):
        try:
            result = openbabel.openbabel_openbabel_protonate_rex(
                input_json,
                config_rex,
                run_spec=RunSpec(target="Bullet"),
                collect=True,
            )
            break
        except TransportConnectionFailed:
            if attempt == 2:
                raise
            time.sleep(2)

    assert not isinstance(result, RunError)
