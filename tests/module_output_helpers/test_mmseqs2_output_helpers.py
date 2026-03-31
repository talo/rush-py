from pathlib import Path

from rush.mmseqs2 import ResultRef


def test_result_ref_fetch_decodes_a3m_objects(monkeypatch):
    monkeypatch.setattr(
        "rush.objects.RushObject.fetch_bytes",
        lambda self, extract=False: f">{self.path}\nSEQUENCE\n".encode(),
    )

    ref = ResultRef.from_raw_output(
        [
            {"path": "0.a3m", "size": 0, "format": "Bin"},
            {"path": "1.a3m", "size": 0, "format": "Bin"},
        ]
    )
    result = ref.fetch()

    assert list(result) == [">0.a3m\nSEQUENCE\n", ">1.a3m\nSEQUENCE\n"]


def test_result_ref_save_saves_a3m_objects(monkeypatch):
    monkeypatch.setattr(
        "rush.objects.RushObject.save",
        lambda self, ext="a3m", **kw: Path(f"/tmp/{self.path}.{ext}"),
    )

    ref = ResultRef.from_raw_output(
        [
            {"path": "0", "size": 0, "format": "Bin"},
            {"path": "1", "size": 0, "format": "Bin"},
        ]
    )
    result = ref.save()

    assert list(result) == [Path("/tmp/0.a3m"), Path("/tmp/1.a3m")]
