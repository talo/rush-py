import json
import tarfile
from io import BytesIO

import pytest
import zstandard as zstd

from rush import exess
from rush.client import _extract_object_archive
from rush.client import RunError


def _make_tar_zst(payload: bytes, filename: str = "output.hdf5") -> bytes:
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        info = tarfile.TarInfo(filename)
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))
    return zstd.ZstdCompressor().compress(archive.getvalue())


def test_fetch_outputs_extracts_hdf5(monkeypatch):
    output_bytes = json.dumps(
        {"schema_version": "0.1.0", "calculation_time": 1.0, "qmmbe": None}
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
        [{"path": "main"}, {"Hdf5": {"path": "exports", "format": "bin"}}]
    )

    assert not isinstance(result, RunError)
    assert result.calc.calculation_time == 1.0
    assert result.exports == b"fake-hdf5"


def test_fetch_outputs_can_skip_extract(monkeypatch):
    output_bytes = json.dumps(
        {"schema_version": "0.1.0", "calculation_time": 1.0, "qmmbe": None}
    ).encode()
    export_bytes = _make_tar_zst(b"fake-hdf5")

    monkeypatch.setattr(
        exess,
        "fetch_object",
        lambda path, extract=False: output_bytes if path == "main" else export_bytes,
    )

    result = exess.fetch_outputs(
        [{"path": "main"}, {"Hdf5": {"path": "exports", "format": "bin"}}],
        extract=False,
    )

    assert not isinstance(result, RunError)
    assert result.exports == export_bytes


def test_fetch_outputs_rejects_unknown_export_wrapper(monkeypatch):
    output_bytes = json.dumps(
        {"schema_version": "0.1.0", "calculation_time": 1.0, "qmmbe": None}
    ).encode()
    monkeypatch.setattr(exess, "fetch_object", lambda path, extract=False: output_bytes)

    with pytest.raises(ValueError, match="Unknown output format"):
        exess.fetch_outputs([{"path": "main"}, {"Csv": {"path": "exports"}}])
