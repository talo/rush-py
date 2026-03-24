import sys
from pathlib import Path

from rush import nnxtb
from rush.client import RunOpts


def test_nnxtb(test_data_dir: Path):
    run = nnxtb.energy(
        test_data_dir / "1kuw_t.json",
        compute_forces=True,
        compute_frequencies=False,  # defaults to False; more expensive to compute
        multiplicity=1,  # also the default (singlet)
        run_opts=RunOpts(
            name="Rush-Py Test NN-xTB 01",
            tags=["rush-py", "test", "1kuw"],
        ),
    )
    result = run.fetch()
    assert isinstance(result, nnxtb.Result)
    print(result, file=sys.stderr)
    print(run.save(), file=sys.stderr)
