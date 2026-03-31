import json
from pathlib import Path

from rush import TRC
from rush import hyper


def test_trc_batch_result_ref_fetch_and_save(monkeypatch):
    fixture_path = Path(__file__).parent.parent / "data" / "hyper" / "valid_trc.json"
    trc_payload = fixture_path.read_text()

    monkeypatch.setattr("rush.objects.RushObject.fetch_json", lambda _self: json.loads(trc_payload))
    monkeypatch.setattr(
        "rush.objects.RushObject.save",
        lambda self, ext="json", **_kw: Path("workspace") / f"{self.path}.{ext}",
    )

    ref = hyper.TRCBatchResultRef.from_raw_output(
        [
            {
                "Ok": [
                    {"Ok": {"path": "solvated", "size": 0, "format": "Json"}},
                    {
                        "Err": {
                            "stage": "Execution",
                            "category": "ToolInput",
                            "message": "bad input",
                            "input_index": 1,
                        }
                    },
                ]
            }
        ]
    )

    fetched = ref.fetch()
    assert isinstance(fetched[0], TRC)
    assert isinstance(fetched[1], hyper.ItemError)
    assert fetched[1].input_index == 1

    saved = ref.save()
    assert saved[0] == Path("workspace/solvated.json")
    assert isinstance(saved[1], hyper.ItemError)


def test_run_result_ref_fetch_and_save(monkeypatch):
    monkeypatch.setattr("rush.objects.RushObject.fetch_bytes", lambda _self: b"binary-data")
    monkeypatch.setattr(
        "rush.objects.RushObject.save",
        lambda self, ext="bin", **_kw: Path("workspace") / f"{self.path}.{ext}",
    )

    ref = hyper.RunResultRef.from_raw_output(
        [
            {
                "Ok": [
                    {
                        "Ok": {
                            "trajectory": {"path": "traj", "size": 0, "format": "Bin"},
                            "checkpoint": {
                                "path": "checkpoint",
                                "size": 0,
                                "format": "Bin",
                            },
                        }
                    },
                    {
                        "Err": {
                            "stage": "OutputParse",
                            "category": "OutputFormat",
                            "message": "empty output",
                            "input_index": 1,
                        }
                    },
                ]
            }
        ]
    )

    fetched = ref.fetch()
    assert fetched[0] == hyper.RunOutput(trajectory=b"binary-data", checkpoint=b"binary-data")
    assert isinstance(fetched[1], hyper.ItemError)

    saved = ref.save()
    assert saved[0] == hyper.RunOutputPaths(
        trajectory=Path("workspace/traj.xtc"),
        checkpoint=Path("workspace/checkpoint.bin"),
    )
    assert isinstance(saved[1], hyper.ItemError)


def test_batch_result_ref_rejects_top_level_error():
    with_error = [{"Err": {"TooManyInputs": {"count": 200, "max": 128}}}]
    try:
        hyper.TRCBatchResultRef.from_raw_output(with_error)
    except ValueError as exc:
        assert "TooManyInputs" in str(exc)
    else:
        raise AssertionError("Expected ValueError for top-level UserError output")
