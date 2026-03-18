from collections.abc import Iterator
from pathlib import Path

from rush import TRCSavedResult
from rush.auto3d import Auto3DResult, Auto3DSavedResult, fetch_outputs, save_outputs
from rush.client import RunError, _json_content_name


def test_fetch_outputs_parses_auto3d_result(monkeypatch):
    fake_trc = object()
    monkeypatch.setattr(
        "rush.auto3d.fetch_object",
        lambda path: b'{"schema_version":"0.1.0"}',
    )
    monkeypatch.setattr("rush.auto3d.from_json", lambda data: fake_trc)

    res = [
        [
            (
                [{"path": "top"}, {"path": "res"}, {"path": "chains"}],
                {
                    "f_max": 0.1,
                    "converged": True,
                    "e_rel_kcal_mol": 1.2,
                    "e_tot_hartrees": -3.4,
                },
            )
        ]
    ]

    output = fetch_outputs(res)

    assert isinstance(output, list)
    first_output = output[0]
    assert isinstance(first_output, Iterator)
    first = list(first_output)
    assert len(first) == 1
    assert isinstance(first[0], Auto3DResult)
    assert first[0].conformer is fake_trc
    assert first[0].stats.converged is True


def test_save_outputs_saves_auto3d_result(monkeypatch):
    monkeypatch.setattr(
        "rush.auto3d.save_object", lambda path: Path(f"/tmp/{path}.json")
    )
    monkeypatch.setattr(
        "rush.auto3d.save_json",
        lambda data, name=None, filepath=None: Path(f"/tmp/{name}.json"),
    )

    res = [
        [
            (
                [{"path": "top"}, {"path": "res"}, {"path": "chains"}],
                {
                    "f_max": 0.1,
                    "converged": True,
                    "e_rel_kcal_mol": 1.2,
                    "e_tot_hartrees": -3.4,
                },
            )
        ]
    ]

    output = save_outputs(res)

    assert isinstance(output, list)
    first_output = output[0]
    assert isinstance(first_output, Iterator)
    first = list(first_output)
    assert len(first) == 1
    assert isinstance(first[0], Auto3DSavedResult)
    assert first[0].conformer == TRCSavedResult(
        topology=Path("/tmp/top.json"),
        residues=Path("/tmp/res.json"),
        chains=Path("/tmp/chains.json"),
    )
    assert first[0].stats == Path(
        f"/tmp/{_json_content_name('auto3d_stats', res[0][0][1])}.json"
    )


def test_auto3d_output_helpers_passthrough_run_id_and_errors():
    err = RunError("Error: auto3d failed")

    assert fetch_outputs("run-id") == "run-id"
    assert save_outputs("run-id") == "run-id"
    assert fetch_outputs(err) is err
    assert save_outputs(err) is err


def test_auto3d_output_helpers_wrap_per_input_errors():
    output = fetch_outputs(["bad smiles"])

    assert isinstance(output, list)
    first_output = output[0]
    assert isinstance(first_output, RunError)
    assert first_output.message == "bad smiles"
