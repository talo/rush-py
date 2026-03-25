from pathlib import Path

from rush import nnxtb
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


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
    assert_run_collects_and_caches(run, nnxtb.ResultRef)

    result = run.fetch()
    assert isinstance(result, nnxtb.Result)
    assert result.energy_mev != 0.0
    assert result.forces_mev_per_angstrom is not None

    saved = run.save()
    assert isinstance(saved, nnxtb.ResultPaths)
    assert saved.output.suffix == ".json"
    assert saved.output.exists()
