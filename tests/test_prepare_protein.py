import json
import sys
from pathlib import Path

from rush_py2.client import RunOpts, save_json, save_object, set_opts
from rush_py2.convert import from_json, to_json
from rush_py2.prepare_protein import prepare_protein


def test_prepare_protein():
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
    res = prepare_protein(
        data_dir / "3FLN.pdb",
        ph=7.0,
        capping_style="always",
        truncation_threshold=5,
        run_opts=RunOpts(
            name="Test prepare-protein 01", tags=["rush-py2", "test", "cdk2"]
        ),
        collect=True,
    )
    trc = from_json(
        (
            save_object(res[0]["path"]),
            save_object(res[1]["path"]),
            save_object(res[2]["path"]),
        ))
    assert 'ACE' in trc.residues.seqs
    assert 'NME' in trc.residues.seqs


if __name__ == "__main__":
    test_prepare_protein()