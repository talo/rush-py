from rush import TRC
from pathlib import Path

from rush.client import RunOpts, set_opts
from rush.prepare_protein import prepare


def test_prepare_protein():
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
    run = prepare(
        data_dir / "3fln_raw.pdb",
        capping_style="always",
        run_opts=RunOpts(
            name="Test prepare-protein 01", tags=["rush-py", "test", "cdk2"]
        ),
    )

    trcs = run.fetch()
    assert isinstance(trcs, list)
    assert len(trcs) >= 1
    trc = trcs[0]
    assert isinstance(trc, TRC)
    residues = trc.residues

    # Ensure the output is capped as requested
    assert "ACE" == residues.seqs[0]
    assert "NME" == residues.seqs[-1]


if __name__ == "__main__":
    test_prepare_protein()
