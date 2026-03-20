import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest
import zstandard as zstd

from rush import exess
from rush.client import _extract_object_archive
from rush.exess_geo_opt import (
    ExessGeoOptResult,
    ExessGeoOptSavedResult,
    ExessGeoOptStep,
)
from rush.exess_geo_opt import fetch_outputs as fetch_geo_opt_outputs
from rush.exess_geo_opt import save_outputs as save_geo_opt_outputs
from rush.exess_qmmm import ExessQMMMResult
from rush.exess_qmmm import fetch_outputs as fetch_qmmm_outputs
from rush.exess_qmmm import save_outputs as save_qmmm_outputs


def _make_tar_zst(payload: bytes, filename: str = "output.hdf5") -> bytes:
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo(filename)
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))
    return zstd.ZstdCompressor().compress(archive.getvalue())


def test_fetch_outputs_extracts_hdf5(monkeypatch):
    output_bytes = json.dumps(
        {"calculation_time": 1.0, "qmmbe": {"method": "RestrictedHF", "nmers": []}}
    ).encode()
    export_bytes = _make_tar_zst(b"fake-hdf5")

    monkeypatch.setattr(
        exess,
        "fetch_object",
        lambda path, extract=False: (
            output_bytes
            if path == "main"
            else _extract_object_archive(export_bytes)
            if extract
            else export_bytes
        ),
    )

    result = exess.fetch_outputs(
        (
            {"path": "main"},
            {"Hdf5": {"path": "exports", "format": "bin"}},
        )
    )

    assert result.calc.calculation_time == 1.0
    assert result.exports == b"fake-hdf5"


def test_fetch_outputs_can_skip_extract(monkeypatch):
    output_bytes = json.dumps(
        {"calculation_time": 1.0, "qmmbe": {"method": "RestrictedHF", "nmers": []}}
    ).encode()
    export_bytes = _make_tar_zst(b"fake-hdf5")

    monkeypatch.setattr(
        exess,
        "fetch_object",
        lambda path, extract=False: output_bytes if path == "main" else export_bytes,
    )

    result = exess.fetch_outputs(
        (
            {"path": "main"},
            {"Hdf5": {"path": "exports", "format": "bin"}},
        ),
        extract=False,
    )

    assert result.exports == export_bytes


def test_fetch_outputs_rejects_unknown_export_wrapper(monkeypatch):
    output_bytes = json.dumps(
        {"calculation_time": 1.0, "qmmbe": {"method": "RestrictedHF", "nmers": []}}
    ).encode()
    monkeypatch.setattr(exess, "fetch_object", lambda path, extract=False: output_bytes)

    with pytest.raises(ValueError, match="Unknown output format"):
        exess.fetch_outputs(
            (
                {"path": "main"},
                {"Csv": {"path": "exports"}},
            )
        )


def test_geo_opt_fetch_outputs(monkeypatch):
    trajectory_json = json.dumps(
        [
            {
                "schema_version": "0.2.0",
                "symbols": ["O", "H", "H"],
                "geometry": [0.0, 0.0, 0.0, 0.7, 0.5, 0.0, -0.7, 0.5, 0.0],
            }
        ]
    ).encode()
    steps_json = json.dumps(
        [
            {
                "total_energy": -76.0,
                "max_gradient_component": 1e-4,
            }
        ]
    ).encode()

    monkeypatch.setattr(
        "rush.exess_geo_opt.fetch_object",
        lambda path: trajectory_json if path == "traj" else steps_json,
    )

    result = fetch_geo_opt_outputs(
        (
            {"path": "traj"},
            {"path": "steps"},
        )
    )

    assert isinstance(result, ExessGeoOptResult)
    assert len(result.trajectory) == 1
    assert result.trajectory[0].geometry == [
        0.0,
        0.0,
        0.0,
        0.7,
        0.5,
        0.0,
        -0.7,
        0.5,
        0.0,
    ]
    assert result.steps == [
        ExessGeoOptStep(total_energy=-76.0, max_gradient_component=1e-4)
    ]


def test_geo_opt_save_outputs(monkeypatch):
    monkeypatch.setattr(
        "rush.exess_geo_opt.save_object",
        lambda path: Path(f"/tmp/{path}.json"),
    )

    result = save_geo_opt_outputs(
        (
            {"path": "traj"},
            {"path": "steps"},
        )
    )

    assert result == ExessGeoOptSavedResult(
        trajectory=Path("/tmp/traj.json"),
        steps=Path("/tmp/steps.json"),
    )


def test_qmmm_fetch_outputs(monkeypatch):
    qmmm_json = json.dumps({"geometries": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]}).encode()

    monkeypatch.setattr(
        "rush.exess_qmmm.fetch_object",
        lambda path: qmmm_json,
    )

    result = fetch_qmmm_outputs({"path": "traj"})

    assert isinstance(result, ExessQMMMResult)
    assert result.geometries == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_qmmm_save_outputs(monkeypatch):
    monkeypatch.setattr(
        "rush.exess_qmmm.save_object",
        lambda path: Path(f"/tmp/{path}.json"),
    )

    result = save_qmmm_outputs({"path": "traj"})

    assert result == Path("/tmp/traj.json")


def test_geo_opt_and_qmmm_output_helpers_reject_invalid_shapes():
    with pytest.raises(ValueError, match="unexpected format"):
        fetch_geo_opt_outputs(({"path": "traj"},))

    with pytest.raises(ValueError, match="unexpected format"):
        save_geo_opt_outputs(({"path": "traj"},))

    with pytest.raises(ValueError, match="unexpected format"):
        fetch_qmmm_outputs({})

    with pytest.raises(ValueError, match="unexpected format"):
        save_qmmm_outputs({})
