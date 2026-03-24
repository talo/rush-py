import json
from pathlib import Path

import pytest

from rush import TRC, TRCPaths
from rush.prepare import ResultRef


def _sample_trc_dict() -> dict:
    data_path = Path(__file__).parent.parent / "data" / "1hsg_MK1_trc.json"
    with data_path.open() as f:
        return json.load(f)[0]


def test_prepare_protein_result_ref_fetch(monkeypatch):
    trc_dict = _sample_trc_dict()

    monkeypatch.setattr(
        "rush._trc.fetch_object",
        lambda path: json.dumps(trc_dict.get(path, {})).encode(),
    )

    ref = ResultRef.from_raw_output(
        [
            [
                {"path": "topology", "size": 0, "format": "Json"},
                {"path": "residues", "size": 0, "format": "Json"},
                {"path": "chains", "size": 0, "format": "Json"},
            ]
        ]
    )
    result = ref.fetch()

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TRC)
    assert result[0].topology.symbols


def test_prepare_protein_result_ref_save(monkeypatch):
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, **kw: Path(f"/tmp/{self.path}.json"),
    )

    ref = ResultRef.from_raw_output(
        [
            [
                {"path": "top", "size": 0, "format": "Json"},
                {"path": "res", "size": 0, "format": "Json"},
                {"path": "chains", "size": 0, "format": "Json"},
            ]
        ]
    )
    result = ref.save()

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] == TRCPaths(
        topology=Path("/tmp/top.json"),
        residues=Path("/tmp/res.json"),
        chains=Path("/tmp/chains.json"),
    )


def test_prepare_protein_multi_model(monkeypatch):
    """Multi-model PDB returns multiple TRCs."""
    trc_dict = _sample_trc_dict()

    monkeypatch.setattr(
        "rush._trc.fetch_object",
        lambda path: json.dumps(trc_dict.get(path.split("_")[0], {})).encode(),
    )

    ref = ResultRef.from_raw_output(
        [
            [
                {"path": "topology_1", "size": 0, "format": "Json"},
                {"path": "residues_1", "size": 0, "format": "Json"},
                {"path": "chains_1", "size": 0, "format": "Json"},
            ],
            [
                {"path": "topology_2", "size": 0, "format": "Json"},
                {"path": "residues_2", "size": 0, "format": "Json"},
                {"path": "chains_2", "size": 0, "format": "Json"},
            ],
        ]
    )
    result = ref.fetch()

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(trc, TRC) for trc in result)


def test_prepare_result_ref_from_raw_output_rejects_invalid():
    with pytest.raises(ValueError, match="should return a non-empty list"):
        ResultRef.from_raw_output("bad")

    with pytest.raises(ValueError, match="expected a list of 3 elements"):
        ResultRef.from_raw_output(
            [
                [
                    {"path": "a", "size": 0, "format": "Json"},
                    {"path": "b", "size": 0, "format": "Json"},
                ]
            ]
        )

    with pytest.raises(ValueError, match="should return a non-empty list"):
        ResultRef.from_raw_output([])
