import json
import sys
from pathlib import Path

from rush_py2.convert import from_json
from rush_py2.trc.merge import merge_trcs


def test_merge_with_paths():
    """Test merging TRCs using file paths."""
    data_dir = Path.cwd() / "tests" / "data"
    seqs = merge_trcs(
        data_dir / "1hsg_MK1_trc.json", data_dir / "1hsg_HOH_trc.json"
    ).residues.seqs
    assert "MK1" in seqs and "HOH" in seqs


def test_merge_with_trcs():
    """Test merging TRCs using TRC objects."""
    data_dir = Path.cwd() / "tests" / "data"
    mk1_trc = from_json(data_dir / "1hsg_MK1_trc.json")
    hoh_trc = from_json(data_dir / "1hsg_HOH_trc.json")
    seqs = merge_trcs(mk1_trc, hoh_trc).residues.seqs
    assert "MK1" in seqs and "HOH" in seqs


if __name__ == "__main__":
    test_merge_with_paths()
    test_merge_with_trcs()
