import json
from pathlib import Path

from rush import TRC, hyper


def _load_fixture(name: str) -> dict:
    fixture_path = Path(__file__).parent.parent / "data" / "hyper" / name
    with fixture_path.open() as f:
        return json.load(f)


def test_solvate_result_ref_fetch_and_save(monkeypatch):
    trc = _load_fixture("valid_trc.json")

    monkeypatch.setattr(
        "rush.hyper._shared.fetch_object",
        lambda _path: json.dumps(trc).encode(),
    )
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, **_kwargs: Path("solvate.json"),
    )

    ref = hyper.SolvateResultRef.from_raw_output(
        [{"Ok": {"path": "solvated", "size": 0, "format": "Json"}}]
    )

    fetched = ref.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], TRC)

    saved = ref.save()
    assert saved == [Path("solvate.json")]


def test_minimize_result_ref_fetch_and_save(monkeypatch):
    trc = _load_fixture("methanol_trc.json")

    monkeypatch.setattr(
        "rush.hyper._shared.fetch_object",
        lambda _path: json.dumps(trc).encode(),
    )
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, **_kwargs: Path("minimized.json"),
    )

    ref = hyper.MinimizeResultRef.from_raw_output(
        [{"Ok": {"path": "minimized", "size": 0, "format": "Json"}}]
    )

    fetched = ref.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], TRC)

    saved = ref.save()
    assert saved == [Path("minimized.json")]


def test_run_result_ref_fetch_and_save(monkeypatch):
    monkeypatch.setattr(
        "rush.hyper._shared.fetch_object",
        lambda path: b"traj-bytes" if path == "traj" else b"checkpoint-bytes",
    )
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, **kwargs: Path(f"{self.path}.{kwargs.get('ext') or 'bin'}"),
    )

    ref = hyper.RunResultRef.from_raw_output(
        [
            {
                "Ok": {
                    "trajectory": {"path": "traj", "size": 10, "format": "Bin"},
                    "checkpoint": {"path": "ckpt", "size": 16, "format": "Bin"},
                }
            }
        ]
    )

    fetched = ref.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], hyper.RunOutput)
    assert fetched[0].trajectory == b"traj-bytes"
    assert fetched[0].checkpoint == b"checkpoint-bytes"

    saved = ref.save()
    assert len(saved) == 1
    assert isinstance(saved[0], hyper.RunOutputPaths)
    assert saved[0].trajectory == Path("traj.xtc")
    assert saved[0].checkpoint == Path("ckpt.cpt")
