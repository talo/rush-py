from pathlib import Path

from rush.client import _json_content_name
from rush.pbsa import PBSAResult, fetch_outputs, save_outputs


def test_fetch_outputs_parses_pbsa_result():
    result = fetch_outputs((-1.0, -0.8, -0.2))

    assert isinstance(result, PBSAResult)
    assert result.solvation_energy == -1.0
    assert result.polar_solvation_energy == -0.8
    assert result.nonpolar_solvation_energy == -0.2


def test_save_outputs_saves_pbsa_result(monkeypatch):
    saved = {}

    def fake_save_json(data, filepath=None, name=None):
        saved["data"] = data
        saved["filepath"] = filepath
        saved["name"] = name
        return Path("/tmp/pbsa_output.json")

    monkeypatch.setattr(
        "rush.pbsa.save_json",
        fake_save_json,
    )

    result = save_outputs((-1.0, -0.8, -0.2))

    assert result == Path("/tmp/pbsa_output.json")
    assert saved["data"] == {
        "solvation_energy": -1.0,
        "polar_solvation_energy": -0.8,
        "nonpolar_solvation_energy": -0.2,
    }
    assert saved["filepath"] is None
    assert saved["name"] == _json_content_name(
        "pbsa_output",
        saved["data"],
    )
