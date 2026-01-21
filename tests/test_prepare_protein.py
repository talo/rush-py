from pathlib import Path

from rush.client import RunOpts, save_object, set_opts
from rush.convert import from_json
from rush.prepare_protein import prepare_protein


def test_prepare_protein():
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
    res = prepare_protein(
        data_dir / "3fln_raw.pdb",
        capping_style="always",
        run_opts=RunOpts(
            name="Test prepare-protein 01", tags=["rush-py", "test", "cdk2"]
        ),
        collect=True,
    )

    # Parse into TRC object
    trc = from_json(tuple(save_object(object["path"]) for object in res))
    if isinstance(trc, list):
        residues = trc[0].residues
    else:
        residues = trc.residues

    # Ensure the output is capped as requested
    assert "ACE" == residues.seqs[0]
    assert "NME" == residues.seqs[-1]


if __name__ == "__main__":
    test_prepare_protein()
