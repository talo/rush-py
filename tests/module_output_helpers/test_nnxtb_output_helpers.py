from pathlib import Path

from rush.nnxtb import Result, ResultPaths, ResultRef


def test_result_ref_fetch_parses_nnxtb_result(monkeypatch):
    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_dict",
        lambda self: {
            "energy_mev": -123.4,
            "forces_mev_per_angstrom": [[1.0, 2.0, 3.0]],
            "frequencies_inv_cm": [100.0, 200.0],
        },
    )

    ref = ResultRef.from_raw_output(
        [{"path": "nnxtb.json", "size": 0, "format": "Json"}]
    )
    result = ref.fetch()

    assert isinstance(result, Result)
    assert result.energy_mev == -123.4
    assert result.forces_mev_per_angstrom == [[1.0, 2.0, 3.0]]
    assert result.frequencies_inv_cm == [100.0, 200.0]


def test_result_ref_save_saves_nnxtb_result(monkeypatch):
    saved_path = Path("/tmp/nnxtb.json")
    monkeypatch.setattr("rush.objects.RushObject.save", lambda self, **kw: saved_path)

    ref = ResultRef.from_raw_output(
        [{"path": "nnxtb.json", "size": 0, "format": "Json"}]
    )
    result = ref.save()

    assert result == ResultPaths(output=saved_path)
