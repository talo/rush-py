from pathlib import Path

from rush import TRC
from rush.client import RunError, RunOpts, set_opts
from rush.prepare_protein import fetch_outputs, prepare_protein


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
    assert not isinstance(res, RunError)

    trc = fetch_outputs(res)
    assert isinstance(trc, TRC)
    residues = trc.residues

    # Ensure the output is capped as requested
    assert "ACE" == residues.seqs[0]
    assert "NME" == residues.seqs[-1]


if __name__ == "__main__":
    test_prepare_protein()
