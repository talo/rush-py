from __future__ import annotations

import json
from pathlib import Path

from rush import TRC, hyper


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _install_fake_save(monkeypatch, tmp_path: Path) -> None:
    def fake_save(self, **kwargs):
        ext = kwargs.get("ext") or "json"
        out = tmp_path / f"{Path(self.path).name}.{ext}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"saved")
        return out

    monkeypatch.setattr("rush.client.RushObject.save", fake_save)


def test_solvate_result_ref_fetch_and_save(monkeypatch, test_data_dir: Path, tmp_path: Path):
    trc_payload = json.dumps(_load_json(test_data_dir / "hyper" / "valid_trc.json")).encode()

    def fake_fetch(path: str):
        assert path == "solvated-ok"
        return trc_payload

    monkeypatch.setattr("rush.hyper._solvate.fetch_object", fake_fetch)
    _install_fake_save(monkeypatch, tmp_path)

    raw = [
        {"Ok": {"path": "solvated-ok", "size": 1, "format": "Json"}},
        {
            "Err": {
                "stage": "Execution",
                "category": "ToolInput",
                "message": "invalid coordinates",
                "input_index": 1,
            }
        },
    ]

    ref = hyper.SolvateResultRef.from_raw_output(raw)
    assert isinstance(ref[0], hyper.SolvateOutputRef)
    assert isinstance(ref[1], hyper.ItemError)

    fetched = ref.fetch()
    assert isinstance(fetched[0], TRC)
    assert isinstance(fetched[1], hyper.ItemError)

    saved = ref.save()
    assert isinstance(saved[0], Path)
    assert saved[0].exists()
    assert isinstance(saved[1], hyper.ItemError)


def test_minimize_result_ref_fetch_and_save(monkeypatch, test_data_dir: Path, tmp_path: Path):
    trc_payload = json.dumps(_load_json(test_data_dir / "hyper" / "methanol_trc.json")).encode()

    def fake_fetch(path: str):
        assert path == "minimize-ok"
        return trc_payload

    monkeypatch.setattr("rush.hyper._minimize.fetch_object", fake_fetch)
    _install_fake_save(monkeypatch, tmp_path)

    raw = [
        {"Ok": {"path": "minimize-ok", "size": 1, "format": "Json"}},
        {
            "Err": {
                "stage": "InputDecode",
                "category": "InvalidInput",
                "message": "broken topology",
                "input_index": 1,
            }
        },
    ]

    ref = hyper.MinimizeResultRef.from_raw_output(raw)
    assert isinstance(ref[0], hyper.MinimizeOutputRef)
    assert isinstance(ref[1], hyper.ItemError)

    fetched = ref.fetch()
    assert isinstance(fetched[0], TRC)
    assert isinstance(fetched[1], hyper.ItemError)

    saved = ref.save()
    assert isinstance(saved[0], Path)
    assert saved[0].exists()
    assert isinstance(saved[1], hyper.ItemError)


def test_run_result_ref_fetch_and_save(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_fetch(path: str):
        calls.append(path)
        if path == "traj-ok":
            return b"xtc-bytes"
        if path == "chk-ok":
            return b"checkpoint-bytes"
        raise AssertionError(f"unexpected fetch path {path}")

    monkeypatch.setattr("rush.hyper._run.fetch_object", fake_fetch)
    _install_fake_save(monkeypatch, tmp_path)

    raw = [
        {
            "Ok": {
                "trajectory": {"path": "traj-ok", "size": 12, "format": "Bin"},
                "checkpoint": {"path": "chk-ok", "size": 8, "format": "Bin"},
            }
        },
        {
            "Err": {
                "stage": "Execution",
                "category": "ToolInput",
                "message": "config rejected",
                "input_index": 1,
            }
        },
    ]

    ref = hyper.RunResultRef.from_raw_output(raw)
    assert isinstance(ref[0], hyper.RunOutputRef)
    assert isinstance(ref[1], hyper.ItemError)

    fetched = ref.fetch()
    assert isinstance(fetched[0], hyper.RunOutput)
    assert fetched[0].trajectory == b"xtc-bytes"
    assert fetched[0].checkpoint == b"checkpoint-bytes"
    assert isinstance(fetched[1], hyper.ItemError)
    assert calls == ["traj-ok", "chk-ok"]

    saved = ref.save()
    assert isinstance(saved[0], hyper.RunOutputPaths)
    assert saved[0].trajectory.exists()
    assert saved[0].trajectory.suffix == ".xtc"
    assert saved[0].checkpoint is not None
    assert saved[0].checkpoint.exists()
    assert saved[0].checkpoint.suffix == ".bin"
    assert isinstance(saved[1], hyper.ItemError)
