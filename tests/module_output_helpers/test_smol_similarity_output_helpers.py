import pytest

from rush.smol_similarity import ExecutionError, Result, ResultRef


def test_result_ref_fetch_parses_successful_item(monkeypatch):
    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_list",
        lambda self: [["CCC", "CCO"], [0.91, 0.73]],
    )

    ref = ResultRef.from_raw_output(
        [
            {
                "Ok": {
                    "path": "object-path",
                    "size": 0,
                    "format": "Json",
                }
            }
        ]
    )

    fetched = ref.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], Result)
    assert fetched[0].smiles == ["CCC", "CCO"]
    assert fetched[0].similarities == [0.91, 0.73]


def test_result_ref_fetch_preserves_item_error():
    ref = ResultRef.from_raw_output(
        [
            {
                "Err": {
                    "stage": "OutputParse",
                    "message": "No valid similarity scores were produced.",
                }
            }
        ]
    )

    fetched = ref.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], ExecutionError)
    assert fetched[0].stage == "OutputParse"


def test_result_ref_rejects_invalid_raw_shape():
    with pytest.raises(ValueError, match="non-empty list"):
        ResultRef.from_raw_output([])

    with pytest.raises(ValueError, match=r"Result\{Ok\|Err\}"):
        ResultRef.from_raw_output(["bad"])
