import base64
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rush import TRCPaths
from rush.boltz import Result, ResultPaths, ResultRef


def _sample_boltz_raw_output():
    """Simulate raw collect_run output: [[sample0_tuple, ...]]."""
    return [
        [
            [
                [
                    {"path": "top", "size": 0, "format": "Json"},
                    {"path": "res", "size": 0, "format": "Json"},
                    {"path": "chains", "size": 0, "format": "Json"},
                ],
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
                {"path": "plddt", "size": 0, "format": "Bin"},
                {"path": "pae", "size": 0, "format": "Bin"},
                {
                    "affinity_pred_value": 1.0,
                    "affinity_probability_binary": 0.1,
                    "affinity_pred_value1": 1.1,
                    "affinity_probability_binary1": 0.2,
                    "affinity_pred_value2": 1.2,
                    "affinity_probability_binary2": 0.3,
                },
            ]
        ]
    ]


def _encode_float_array(values: list[float]) -> str:
    raw = np.asarray(values, dtype=np.float32).tobytes()
    return base64.b64encode(raw).decode()


def test_result_ref_fetch_parses_boltz_result(monkeypatch):
    fake_trc = object()

    def fake_fetch_dict(self):
        payloads = {
            "plddt": (
                '{"shape":[3],"data":"%s"}' % _encode_float_array([1.0, 2.0, 3.0])
            ),
            "pae": (
                '{"shape":[3,3],"data":"%s"}'
                % _encode_float_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            ),
        }
        if self.path in payloads:
            return __import__("json").loads(payloads[self.path])
        return {}

    monkeypatch.setattr("rush.objects.RushObject.fetch_dict", fake_fetch_dict)
    monkeypatch.setattr("rush.objects.fetch_object", lambda path, extract=False: b"{}")
    monkeypatch.setattr("rush.objects.from_json", lambda data: fake_trc)

    ref = ResultRef.from_raw_output(_sample_boltz_raw_output())
    output = list(ref.fetch())

    assert len(output) == 1
    assert isinstance(output[0], Result)
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


def test_result_ref_save_saves_boltz_result(monkeypatch):
    saved_json_names = []

    monkeypatch.setattr(
        "rush.objects.RushObject.save",
        lambda self, **kw: Path(f"/tmp/{self.path}.json"),
    )

    def fake_save_json(data, filepath=None, name=None):
        saved_json_names.append(name)
        return Path(f"/tmp/{name}.json")

    monkeypatch.setattr("rush.boltz.save_json", fake_save_json)

    ref = ResultRef.from_raw_output(_sample_boltz_raw_output())
    output = list(ref.save())

    assert len(output) == 1
    assert isinstance(output[0], ResultPaths)
    assert output[0].model == TRCPaths(
        topology=Path("/tmp/top.json"),
        residues=Path("/tmp/res.json"),
        chains=Path("/tmp/chains.json"),
    )
    assert output[0].plddt == Path("/tmp/plddt.json")
    assert output[0].pae == Path("/tmp/pae.json")
    assert output[0].affinities is not None
    assert len(saved_json_names) == 2
    assert saved_json_names[0].startswith("boltz_metrics_")
    assert saved_json_names[1].startswith("boltz_affinities_")


def test_boltz_result_ref_rejects_malformed_output():
    # Raw output missing the affinities element in the tuple — now caught at parse time
    bad_raw: list[Any] = [
        [
            [
                [
                    {"path": "top", "size": 0, "format": "Json"},
                    {"path": "res", "size": 0, "format": "Json"},
                    {"path": "chains", "size": 0, "format": "Json"},
                ],
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
                {"path": "plddt", "size": 0, "format": "Bin"},
                {"path": "pae", "size": 0, "format": "Bin"},
            ],
        ]
    ]

    with pytest.raises(ValueError, match="not enough values to unpack"):
        ResultRef.from_raw_output(bad_raw)
