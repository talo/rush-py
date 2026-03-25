from rush import TRC
from pathlib import Path

from rush import prepare
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_prepare_protein(test_data_dir: Path):
    run = prepare.protein(
        test_data_dir / "3fln_raw.pdb",
        capping_style="always",
        run_opts=RunOpts(
            name="Test prepare-protein 01", tags=["rush-py", "test", "cdk2"]
        ),
    )
    assert_run_collects_and_caches(run, prepare.ResultRef)

    trcs = run.fetch()
    assert isinstance(trcs, list)
    assert len(trcs) >= 1
    trc = trcs[0]
    assert isinstance(trc, TRC)
    residues = trc.residues

    # Ensure the output is capped as requested
    assert "ACE" == residues.seqs[0]
    assert "NME" == residues.seqs[-1]

    saved = run.save()
    assert len(saved) == len(trcs)
    for saved_model in saved:
        assert saved_model.topology.exists()
        assert saved_model.residues.exists()
        assert saved_model.chains.exists()
