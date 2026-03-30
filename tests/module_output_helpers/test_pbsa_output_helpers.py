from pathlib import Path

from rush.objects import _json_content_name
from rush.pbsa import Result, ResultPaths, ResultRef


def test_result_ref_fetch_parses_pbsa_result():
    ref = ResultRef.from_raw_output([-1.0, -0.8, -0.2])
    result = ref.fetch()

    assert isinstance(result, Result)
    assert result.solvation_energy == -1.0
    assert result.polar_solvation_energy == -0.8
    assert result.nonpolar_solvation_energy == -0.2


def test_result_ref_save_saves_pbsa_result(monkeypatch):
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

    ref = ResultRef.from_raw_output([-1.0, -0.8, -0.2])
    result = ref.save()

    assert result == ResultPaths(output=Path("/tmp/pbsa_output.json"))
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
