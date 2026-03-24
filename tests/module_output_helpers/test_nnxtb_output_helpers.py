import json
from pathlib import Path

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
