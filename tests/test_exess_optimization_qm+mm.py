from pathlib import Path

from rush import exess
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_optimization_qm_mm(test_data_dir: Path):
    run = exess.optimization(
        (test_data_dir / "6a5j_t.json", test_data_dir / "6a5j_r.json"),
        max_iters=100,
        optimization_keywords=exess.OptimizationKeywords(
            coordinate_system="Cartesian",
            algorithm="LBFGS",
            lbfgs_keywords=exess.LBFGSKeywords(),
        ),
        ksdft_keywords=exess.KSDFTKeywords(
            functional="B3LYP",
            grid=exess.XCGridParameters(
                radial_quad="TreutlerAldrichs",
                pruning_scheme="Treutler",
                resolution=exess.DefaultGridResolution("TreutlerGM5"),
                batching=exess.SpaceFillingBatching(),
            ),
        ),
        standard_orientation="None",
        qm_fragments=[0],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Optimization 04: QM+MM",
            tags=["rush-py", "test", "6a5j", "QM+MM"],
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
