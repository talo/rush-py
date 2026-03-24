import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, set_opts


def test_exess_optimization_qm():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    run = exess.optimization(
        data_dir / "benzene_t.json",
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
    print(run, file=sys.stderr)
    fetched = run.fetch()
    assert isinstance(fetched, exess.OptimizationResult)
    assert fetched.trajectory
    assert fetched.steps

    saved = run.save()
    assert isinstance(saved, exess.OptimizationResultPaths)


if __name__ == "__main__":
    test_exess_optimization_qm()
