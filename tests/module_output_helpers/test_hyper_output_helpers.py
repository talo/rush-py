import json
from pathlib import Path

from rush import TRC, hyper


_SAMPLE_TRC = {
    "topology": {
        "schema_version": "0.2.0",
        "symbols": ["O", "H", "H"],
        "geometry": [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0],
        "labels": None,
        "partial_charges": None,
        "formal_charges": None,
        "connectivity": None,
        "stereochemistry": None,
        "velocities": None,
        "fragments": None,
        "fragment_formal_charges": None,
        "fragment_partial_charges": None,
        "fragment_multiplicities": None,
    },
    "residues": {
        "residues": [[0, 1, 2]],
        "seqs": ["HOH"],
        "seq_ns": [1],
        "insertion_codes": [""],
        "labeled": None,
        "labels": None,
    },
    "chains": {
        "chains": [[0]],
        "alpha_helices": None,
        "beta_sheets": None,
        "labeled": None,
        "labels": None,
    },
}


def test_solvate_result_ref_fetches_and_saves_trc(monkeypatch, tmp_path: Path):
    raw = [{"Ok": {"path": "solvated-1", "size": 0, "format": "Json"}}]

    monkeypatch.setattr(
        "rush.hyper._common.fetch_object",
        lambda path: json.dumps(_SAMPLE_TRC).encode(),
    )
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, ext="json", **kw: tmp_path / f"{self.path}.{ext}",
    )

    ref = hyper.SolvateResultRef.from_raw_output(raw)

    fetched = ref.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], TRC)

    saved = ref.save()
    assert saved == [tmp_path / "solvated-1.json"]


def test_minimize_result_ref_parses_item_error():
    raw = [
        {
            "Err": {
                "stage": "InputDecode",
                "category": "InvalidInput",
                "message": "broken",
                "input_index": 0,
            }
        }
    ]

    ref = hyper.MinimizeResultRef.from_raw_output(raw)
    outputs = ref.fetch()

    assert len(outputs) == 1
    assert isinstance(outputs[0], hyper.ItemError)
    assert outputs[0].message == "broken"


def test_run_result_ref_fetches_artifacts_and_preserves_errors(monkeypatch, tmp_path: Path):
    raw = [
        {
            "Ok": {
                "trajectory": {"path": "traj-1", "size": 3, "format": "Bin"},
                "checkpoint": {"path": "chk-1", "size": 2, "format": "Bin"},
            }
        },
        {
            "Err": {
                "stage": "Execution",
                "category": "ToolInput",
                "message": "bad config",
                "input_index": 1,
            }
        },
    ]

    monkeypatch.setattr(
        "rush.hyper._common.fetch_object",
        lambda path: b"XTC" if path == "traj-1" else b"CK",
    )
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, ext="bin", **kw: tmp_path / f"{self.path}.{ext}",
    )

    ref = hyper.RunResultRef.from_raw_output(raw)

    fetched = ref.fetch()
    assert isinstance(fetched[0], hyper.HyperRunOutput)
    assert fetched[0].trajectory == b"XTC"
    assert fetched[0].checkpoint == b"CK"
    assert isinstance(fetched[1], hyper.ItemError)

    saved = ref.save()
    assert isinstance(saved[0], hyper.HyperRunOutputPaths)
    assert saved[0].trajectory == tmp_path / "traj-1.xtc"
    assert saved[0].checkpoint == tmp_path / "chk-1.bin"
    assert isinstance(saved[1], hyper.ItemError)
