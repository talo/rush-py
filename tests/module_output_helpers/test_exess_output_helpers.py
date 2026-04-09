import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import zstandard as zstd

from rush.exess import (
    OptimizationResult,
    OptimizationResultPaths,
    OptimizationResultRef,
    OptimizationStep,
    QMMMResult,
    QMMMResultPaths,
    QMMMResultRef,
    ResultRef,
)
from rush.objects import _extract_object_archive


def _make_tar_zst(payload: bytes, filename: str = "output.hdf5") -> bytes:
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo(filename)
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))
    return zstd.ZstdCompressor().compress(archive.getvalue())


def test_result_ref_fetch_extracts_hdf5(monkeypatch):
    output_bytes = json.dumps(
        {"calculation_time": 1.0, "qmmbe": {"method": "RestrictedHF", "nmers": []}}
    ).encode()
    export_bytes = _make_tar_zst(b"fake-hdf5")

    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_dict",
        lambda self: json.loads(output_bytes),
    )
    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_bytes",
        lambda self, extract=False: (
            _extract_object_archive(export_bytes) if extract else export_bytes
        ),
    )

    ref = ResultRef.from_raw_output(
        [
            {"path": "main", "size": 0, "format": "Json"},
            {"Hdf5": {"path": "exports", "size": 0, "format": "Bin"}},
        ]
    )
    result = ref.fetch()

    assert result.calc.calculation_time == 1.0
    assert result.exports == b"fake-hdf5"


def test_result_ref_fetch_can_skip_extract(monkeypatch):
    output_bytes = json.dumps(
        {"calculation_time": 1.0, "qmmbe": {"method": "RestrictedHF", "nmers": []}}
    ).encode()
    export_bytes = _make_tar_zst(b"fake-hdf5")

    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_dict",
        lambda self: json.loads(output_bytes),
    )
    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_bytes",
        lambda self, extract=False: export_bytes,
    )

    ref = ResultRef.from_raw_output(
        [
            {"path": "main", "size": 0, "format": "Json"},
            {"Hdf5": {"path": "exports", "size": 0, "format": "Bin"}},
        ]
    )
    result = ref.fetch(extract=False)

    assert result.exports == export_bytes


def test_result_ref_rejects_unknown_export_wrapper(monkeypatch):
    output_bytes = json.dumps(
        {"calculation_time": 1.0, "qmmbe": {"method": "RestrictedHF", "nmers": []}}
    ).encode()
    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_dict",
        lambda self: json.loads(output_bytes),
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        ResultRef.from_raw_output(
            [
                {"path": "main", "size": 0, "format": "Json"},
                {"Csv": {"path": "exports"}},
            ]
        )


def test_optimization_ref_fetch(monkeypatch):
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
        "rush.objects.RushObject.fetch_list",
        lambda self: json.loads(trajectory_json if self.path == "traj" else steps_json),
    )

    ref = OptimizationResultRef.from_raw_output(
        [
            {"path": "traj", "size": 0, "format": "Json"},
            {"path": "steps", "size": 0, "format": "Json"},
        ]
    )
    result = ref.fetch()

    assert isinstance(result, OptimizationResult)
    assert len(result.trajectory) == 1
    np.testing.assert_allclose(
        result.trajectory[0].geometry,
        [[0.0, 0.0, 0.0], [0.7, 0.5, 0.0], [-0.7, 0.5, 0.0]],
        atol=1e-6,
    )
    assert result.steps == [
        OptimizationStep(total_energy=-76.0, max_gradient_component=1e-4)
    ]


def test_optimization_ref_save(monkeypatch):
    monkeypatch.setattr(
        "rush.objects.RushObject.save",
        lambda self, **kw: Path(f"/tmp/{self.path}.json"),
    )

    ref = OptimizationResultRef.from_raw_output(
        [
            {"path": "traj", "size": 0, "format": "Json"},
            {"path": "steps", "size": 0, "format": "Json"},
        ]
    )
    result = ref.save()

    assert result == OptimizationResultPaths(
        trajectory=Path("/tmp/traj.json"),
        steps=Path("/tmp/steps.json"),
    )


def test_qmmm_ref_fetch(monkeypatch):
    qmmm_json = json.dumps({"geometries": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]}).encode()

    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_dict",
        lambda self: json.loads(qmmm_json),
    )

    ref = QMMMResultRef.from_raw_output({"path": "traj", "size": 0, "format": "Json"})
    result = ref.fetch()

    assert isinstance(result, QMMMResult)
    assert len(result.geometries) == 2
    np.testing.assert_allclose(result.geometries[0], [[0.0, 1.0, 2.0]], atol=1e-6)
    np.testing.assert_allclose(result.geometries[1], [[3.0, 4.0, 5.0]], atol=1e-6)


def test_qmmm_ref_save(monkeypatch):
    monkeypatch.setattr(
        "rush.objects.RushObject.save",
        lambda self, **kw: Path(f"/tmp/{self.path}.json"),
    )

    ref = QMMMResultRef.from_raw_output({"path": "traj", "size": 0, "format": "Json"})
    result = ref.save()

    assert result == QMMMResultPaths(output=Path("/tmp/traj.json"))


def test_result_ref_rejects_invalid_shapes():
    with pytest.raises(
        ValueError, match="optimization should return exactly 2 outputs"
    ):
        OptimizationResultRef.from_raw_output(
            [{"path": "traj", "size": 0, "format": "Json"}]
        )

    with pytest.raises(ValueError, match="unexpected format"):
        QMMMResultRef.from_raw_output({})
