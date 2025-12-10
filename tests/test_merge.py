import json
import sys
from pathlib import Path

from rush_py2.convert import from_json
from rush_py2.trc.merge import merge_trcs

if __name__ == "__main__":
    data_dir = Path.cwd() / "tests" / "data"

    # Test with Paths (could also be strings)
    seqs = merge_trcs(
        data_dir / "1hsg_MK1_trc.json", data_dir / "1hsg_HOH_trc.json"
    ).residues.seqs
    assert "MK1" in seqs and "HOH" in seqs

    # Test with TRCs
    mk1_trc = from_json(data_dir / "1hsg_MK1_trc.json")
    hoh_trc = from_json(data_dir / "1hsg_HOH_trc.json")
    seqs = merge_trcs(mk1_trc, hoh_trc).residues.seqs
    assert "MK1" in seqs and "HOH" in seqs
