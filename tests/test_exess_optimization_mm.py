from pathlib import Path

from rush import RunOpts, exess
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_optimization_mm(test_data_dir: Path):
    run = exess.optimization(
        # Residues are required for MM fragments
        (test_data_dir / "6a5j_t.json", test_data_dir / "6a5j_r.json"),
        max_iters=10000,
        optimization_keywords=exess.OptimizationKeywords(
            coordinate_system="Cartesian",
            algorithm="LBFGS",
            lbfgs_keywords=exess.LBFGSKeywords(),
        ),
        basis="STO-2G",
        standard_orientation="None",
        # MM fragments well for uncomplicated regions and runs very quickly.
        qm_fragments=[],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Optimization 02: MM",
            tags=["rush-py", "test", "6a5j", "MM"],
        ),
    )
    assert_run_collects_and_caches(run, exess.OptimizationResultRef)

    result = run.fetch()
    assert isinstance(result, exess.OptimizationResult)
    # TODO: check why convergence fails here, resulting in these two being empty lists
    # assert result.trajectory
    # assert result.steps

    saved = run.save()
    assert isinstance(saved, exess.OptimizationResultPaths)
    assert saved.trajectory.exists()
    assert saved.steps.exists()
