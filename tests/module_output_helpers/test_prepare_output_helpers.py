import json
from pathlib import Path

import pytest

from rush import TRC, TRCSavedResult, from_json
from rush.prepare_complex import fetch_outputs as fetch_prepare_complex_outputs
from rush.prepare_complex import save_outputs as save_prepare_complex_outputs
from rush.prepare_protein import fetch_outputs as fetch_prepare_protein_outputs
from rush.prepare_protein import save_outputs as save_prepare_protein_outputs


def _sample_trc_dict() -> dict:
    data_path = Path(__file__).parent.parent / "data" / "1hsg_MK1_trc.json"
    with data_path.open() as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def test_prepare_protein_fetch_outputs(monkeypatch):
    trc_dict = _sample_trc_dict()

    monkeypatch.setattr(
        "rush.prepare_protein.fetch_object",
        lambda path: json.dumps(
            {
                "top": trc_dict["topology"],
                "res": trc_dict["residues"],
                "chains": trc_dict["chains"],
            }[path]
        ).encode(),
    )

    result = fetch_prepare_protein_outputs(
        ({"path": "top"}, {"path": "res"}, {"path": "chains"})
    )

    assert isinstance(result, TRC)
    assert result.topology.symbols
    assert result.residues.residues
    assert result.chains.chains


def test_prepare_protein_save_outputs(monkeypatch):
    monkeypatch.setattr(
        "rush.prepare_protein.save_object",
        lambda path: Path(f"/tmp/{path}.json"),
    )

    result = save_prepare_protein_outputs(
        ({"path": "top"}, {"path": "res"}, {"path": "chains"})
    )

    assert result == TRCSavedResult(
        topology=Path("/tmp/top.json"),
        residues=Path("/tmp/res.json"),
        chains=Path("/tmp/chains.json"),
    )


def test_prepare_complex_fetch_outputs(monkeypatch):
    trc_dict = _sample_trc_dict()

    monkeypatch.setattr(
        "rush.prepare_complex.fetch_trc_output",
        lambda res: from_json(trc_dict),
    )

    result = fetch_prepare_complex_outputs(
        ({"path": "top"}, {"path": "res"}, {"path": "chains"})
    )

    assert isinstance(result, TRC)
    assert result.topology.symbols


def test_prepare_complex_save_outputs(monkeypatch):
    monkeypatch.setattr(
        "rush.prepare_complex.save_trc_output",
        lambda res: TRCSavedResult(
            topology=Path("/tmp/top.json"),
            residues=Path("/tmp/res.json"),
            chains=Path("/tmp/chains.json"),
        ),
    )

    result = save_prepare_complex_outputs(
        ({"path": "top"}, {"path": "res"}, {"path": "chains"})
    )

    assert result == TRCSavedResult(
        topology=Path("/tmp/top.json"),
        residues=Path("/tmp/res.json"),
        chains=Path("/tmp/chains.json"),
    )


def test_prepare_output_helpers_reject_invalid_shapes():
    bad_outputs = ({"path": object()}, {"path": object()})

    with pytest.raises(ValueError, match="unexpected format"):
        fetch_prepare_protein_outputs(bad_outputs)

    with pytest.raises(ValueError, match="unexpected format"):
        save_prepare_protein_outputs(bad_outputs)

    with pytest.raises(ValueError, match="unexpected format"):
        fetch_prepare_complex_outputs(bad_outputs)

    with pytest.raises(ValueError, match="unexpected format"):
        save_prepare_complex_outputs(bad_outputs)
