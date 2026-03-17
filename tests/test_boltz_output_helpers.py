import base64
from pathlib import Path
from typing import Any, cast

import numpy as np
from rush.boltz import (
    BoltzResult,
    BoltzSavedResult,
    fetch_outputs,
    save_outputs,
)
from rush.client import RunError


def _sample_boltz_output():
    return [
        (
            [{"path": "top"}, {"path": "res"}, {"path": "chains"}],
            {
                "confidence_score": 0.91,
                "ptm": 0.92,
                "iptm": 0.93,
                "ligand_iptm": 0.94,
                "protein_iptm": 0.95,
                "complex_plddt": 0.96,
                "complex_iplddt": 0.97,
                "complex_pde": 0.98,
                "complex_ipde": 0.99,
            },
            {"path": "plddt"},
            {"path": "pae"},
            {
                "affinity_pred_value": 1.0,
                "affinity_probability_binary": 0.1,
                "affinity_pred_value1": 1.1,
                "affinity_probability_binary1": 0.2,
                "affinity_pred_value2": 1.2,
                "affinity_probability_binary2": 0.3,
            },
        )
    ]


def _encode_float_array(values: list[float]) -> str:
    raw = np.asarray(values, dtype=np.float32).tobytes()
    return base64.b64encode(raw).decode()


def test_fetch_outputs_parses_boltz_result(monkeypatch):
    fake_trc = object()

    def fake_fetch_object(path):
        payloads = {
            "top": '{"atoms":[]}',
            "res": '{"ids":[]}',
            "chains": '{"ids":[]}',
            "plddt": (
                '{"shape":[3],"data":"%s"}' % _encode_float_array([1.0, 2.0, 3.0])
            ).encode(),
            "pae": (
                '{"shape":[3,3],"data":"%s"}'
                % _encode_float_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            ).encode(),
        }
        return payloads[path]

    monkeypatch.setattr("rush.boltz.fetch_object", fake_fetch_object)
    monkeypatch.setattr("rush.boltz.from_json", lambda data: fake_trc)

    output = fetch_outputs(_sample_boltz_output())

    assert isinstance(output, list)
    assert isinstance(output[0], BoltzResult)
    assert output[0].model is fake_trc
    assert output[0].metrics.confidence_score == 0.91
    assert isinstance(output[0].plddt, np.ndarray)
    assert output[0].plddt.shape == (3,)
    assert np.array_equal(output[0].plddt, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert output[0].pae.shape == (3, 3)
    assert np.array_equal(
        output[0].pae,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        ),
    )
    assert output[0].affinities is not None
    assert output[0].affinities.affinity_pred_value2 == 1.2


def test_save_outputs_saves_boltz_result(monkeypatch):
    saved_json_names = []

    monkeypatch.setattr(
        "rush.boltz.save_object",
        lambda path: Path(f"/tmp/{path}.json"),
    )

    def fake_save_json(data, filepath=None, name=None):
        saved_json_names.append(name)
        return Path(f"/tmp/{name}.json")

    monkeypatch.setattr("rush.boltz.save_json", fake_save_json)

    output = save_outputs(_sample_boltz_output())

    assert isinstance(output, list)
    assert isinstance(output[0], BoltzSavedResult)
    assert output[0].model == (
        Path("/tmp/top.json"),
        Path("/tmp/res.json"),
        Path("/tmp/chains.json"),
    )
    assert output[0].plddt == Path("/tmp/plddt.json")
    assert output[0].pae == Path("/tmp/pae.json")
    assert output[0].affinities is not None
    assert len(saved_json_names) == 2
    assert saved_json_names[0].startswith("boltz_metrics_")
    assert saved_json_names[1].startswith("boltz_affinities_")


def test_boltz_output_helpers_passthrough_run_id_and_errors():
    err = RunError("Error: boltz failed")

    assert fetch_outputs("run-id") == "run-id"
    assert save_outputs("run-id") == "run-id"
    assert fetch_outputs(err) is err
    assert save_outputs(err) is err


def test_boltz_output_helpers_reject_unexpected_shape():
    bad_input = cast(Any, {"path": "bad"})
    output = fetch_outputs(bad_input)

    assert isinstance(output, RunError)
    assert "unexpected format" in output.message
