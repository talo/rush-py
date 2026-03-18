from pathlib import Path
from typing import Any, cast

from rush.client import RunError
from rush.mmseqs2 import fetch_outputs, save_outputs


def test_fetch_outputs_decodes_a3m_objects(monkeypatch):
    monkeypatch.setattr(
        "rush.mmseqs2.fetch_object",
        lambda path: f">{path}\nSEQUENCE\n".encode(),
    )

    output = fetch_outputs([{"path": "0.a3m"}, {"path": "1.a3m"}])

    assert output == [">0.a3m\nSEQUENCE\n", ">1.a3m\nSEQUENCE\n"]


def test_save_outputs_saves_a3m_objects(monkeypatch):
    monkeypatch.setattr(
        "rush.mmseqs2.save_object",
        lambda path, type="bin", ext="a3m": Path(f"/tmp/{path}.{ext}"),
    )

    output = save_outputs([{"path": "0"}, {"path": "1"}])

    assert output == [Path("/tmp/0.a3m"), Path("/tmp/1.a3m")]


def test_mmseqs2_output_helpers_passthrough_run_id_and_errors():
    err = RunError("Error: mmseqs2 failed")

    assert fetch_outputs("run-id") == "run-id"
    assert save_outputs("run-id") == "run-id"
    assert fetch_outputs(err) is err
    assert save_outputs(err) is err


def test_mmseqs2_output_helpers_reject_unexpected_shape():
    bad_input = cast(Any, {"path": "0.a3m"})
    output = fetch_outputs(bad_input)

    assert isinstance(output, RunError)
    assert "unexpected format" in output.message
