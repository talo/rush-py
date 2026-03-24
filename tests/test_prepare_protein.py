from rush import TRC
from pathlib import Path

from rush.client import RunOpts
from rush.prepare import protein as prepare


def test_prepare_protein(test_data_dir: Path):
    run = prepare(
        test_data_dir / "3fln_raw.pdb",
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
