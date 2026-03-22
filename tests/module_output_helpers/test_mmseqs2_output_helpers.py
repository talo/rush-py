from pathlib import Path

from rush.mmseqs2 import Result, ResultPaths, ResultRef


def test_result_ref_fetch_decodes_a3m_objects(monkeypatch):
    monkeypatch.setattr(
        "rush.mmseqs2.fetch_object",
        lambda path: f">{path}\nSEQUENCE\n".encode(),
    )

    ref = ResultRef.from_raw_output(
        [
            {"path": "0.a3m", "size": 0, "format": "Bin"},
            {"path": "1.a3m", "size": 0, "format": "Bin"},
        ]
    )
    result = ref.fetch()

    assert isinstance(result, Result)
    assert result.a3m_texts == [">0.a3m\nSEQUENCE\n", ">1.a3m\nSEQUENCE\n"]


def test_result_ref_save_saves_a3m_objects(monkeypatch):
    monkeypatch.setattr(
        "rush.client.RushObject.save",
        lambda self, ext="a3m", **kw: Path(f"/tmp/{self.path}.{ext}"),
    )

    ref = ResultRef.from_raw_output(
        [
            {"path": "0", "size": 0, "format": "Bin"},
            {"path": "1", "size": 0, "format": "Bin"},
        ]
    )
    result = ref.save()

    assert isinstance(result, ResultPaths)
    assert result.a3m_files == [Path("/tmp/0.a3m"), Path("/tmp/1.a3m")]
