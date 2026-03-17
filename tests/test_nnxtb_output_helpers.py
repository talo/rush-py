import json
from pathlib import Path

import pytest

from rush.client import RunError
from rush.nnxtb import NnxtbResult, fetch_outputs, save_outputs


def test_fetch_outputs_parses_nnxtb_result(monkeypatch):
    monkeypatch.setattr(
        "rush.nnxtb.fetch_object",
        lambda path: json.dumps(
            {
                "energy_mev": -123.4,
                "forces_mev_per_angstrom": [[1.0, 2.0, 3.0]],
                "frequencies_inv_cm": [100.0, 200.0],
            }
        ).encode(),
    )

    result = fetch_outputs({"path": "nnxtb.json"})

    assert isinstance(result, NnxtbResult)
    assert result.energy_mev == -123.4
    assert result.forces_mev_per_angstrom == [[1.0, 2.0, 3.0]]
    assert result.frequencies_inv_cm == [100.0, 200.0]


def test_save_outputs_saves_nnxtb_result(monkeypatch):
    saved_path = Path("/tmp/nnxtb.json")
    monkeypatch.setattr("rush.nnxtb.save_object", lambda path: saved_path)

    result = save_outputs({"path": "nnxtb.json"})

    assert result == saved_path


def test_nnxtb_output_helpers_passthrough_run_error():
    err = RunError("Error: nnxtb failed")

    assert fetch_outputs(err) is err
    assert save_outputs(err) is err


def test_nnxtb_output_helpers_reject_multiple_outputs():
    with pytest.raises(ValueError, match="exactly 1 output"):
        fetch_outputs([{"path": "a.json"}, {"path": "b.json"}])
