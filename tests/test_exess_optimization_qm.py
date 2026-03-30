from pathlib import Path

from rush import exess
from rush import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_optimization_qm(test_data_dir: Path):
    run = exess.optimization(
        test_data_dir / "benzene_t.json",
        max_iters=100,
        optimization_keywords=exess.OptimizationKeywords(),
        method="RestrictedRIMP2",
        basis="cc-pVDZ",
        aux_basis="cc-pVDZ-RIFIT",
        standard_orientation="None",
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Optimization 03: QM",
            tags=["rush-py", "test", "benzene", "QM"],
        ),
    )
    assert_run_collects_and_caches(run, exess.OptimizationResultRef)

    result = run.fetch()
    assert isinstance(result, exess.OptimizationResult)
    assert result.trajectory
    assert result.steps

    saved = run.save()
    assert isinstance(saved, exess.OptimizationResultPaths)
    assert saved.trajectory.exists()
    assert saved.steps.exists()
