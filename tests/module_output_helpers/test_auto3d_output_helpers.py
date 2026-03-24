from collections.abc import Iterator
from pathlib import Path

import pytest

from rush import TRCPaths
from rush.auto3d import Result, ResultPaths, ResultRef
from rush.client import _json_content_name


def _sample_raw():
    """Simulate raw collect_run output for one SMILES with one conformer."""
    return [
        [
            [
                [
                    {"path": "top", "size": 0, "format": "Json"},
                    {"path": "res", "size": 0, "format": "Json"},
                    {"path": "chains", "size": 0, "format": "Json"},
                ],
                {
                    "f_max": 0.1,
                    "converged": True,
                    "e_rel_kcal_mol": 1.2,
                    "e_tot_hartrees": -3.4,
                },
            ]
        ]
    ]


def test_fetch_parses_auto3d_result(monkeypatch):
    fake_trc = object()
    monkeypatch.setattr(
        "rush._trc.fetch_object",
        lambda path: b'{"schema_version":"0.1.0"}',
    )
    monkeypatch.setattr("rush._trc.from_json", lambda data: fake_trc)

    ref = ResultRef.from_raw_output(_sample_raw())
    output = ref.fetch()

    assert isinstance(output, list)
    first_output = output[0]
    assert isinstance(first_output, Iterator)
    first = list(first_output)
    assert len(first) == 1
    assert isinstance(first[0], Result)
    assert first[0].conformer is fake_trc
    assert first[0].stats.converged is True


def test_save_saves_auto3d_result(monkeypatch):
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, **kw: Path(f"/tmp/{self.path}.json"),
    )
    monkeypatch.setattr(
        "rush.auto3d.save_json",
        lambda data, name=None, filepath=None: Path(f"/tmp/{name}.json"),
    )

    raw = _sample_raw()
    ref = ResultRef.from_raw_output(raw)
    output = ref.save()

    assert isinstance(output, list)
    first_output = output[0]
    assert isinstance(first_output, Iterator)
    first = list(first_output)
    assert len(first) == 1
    assert isinstance(first[0], ResultPaths)
    assert first[0].conformer == TRCPaths(
        topology=Path("/tmp/top.json"),
        residues=Path("/tmp/res.json"),
        chains=Path("/tmp/chains.json"),
    )
    expected_stats = {
        "f_max": 0.1,
        "converged": True,
        "e_rel_kcal_mol": 1.2,
        "e_tot_hartrees": -3.4,
    }
    assert first[0].stats == Path(
        f"/tmp/{_json_content_name('auto3d_stats', expected_stats)}.json"
    )


def test_auto3d_output_helpers_wrap_per_input_errors():
    ref = ResultRef.from_raw_output(["bad smiles"])
    output = ref.fetch()

    assert isinstance(output, list)
    first_output = output[0]
    assert first_output == "bad smiles"


def test_auto3d_output_helpers_reject_malformed_conformer_payload():
    raw = [
        [
            [
                [
                    {"path": "top", "size": 0, "format": "Json"},
                    {"path": "res", "size": 0, "format": "Json"},
                    {"path": "chains", "size": 0, "format": "Json"},
                ],
                {
                    "f_max": 0.1,
                    "converged": True,
                    "e_rel_kcal_mol": 1.2,
                    "e_tot_hartrees": -3.4,
                },
                "extra",
            ]
        ]
    ]

    with pytest.raises(ValueError, match="too many values to unpack"):
        ResultRef.from_raw_output(raw)
