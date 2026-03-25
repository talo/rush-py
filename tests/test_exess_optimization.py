from pathlib import Path

import pytest

from rush import exess
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


@pytest.mark.skip(reason="ML regions are disabled upstream for now.")
def test_exess_optimization(test_data_dir: Path):
    # Setting `standard_orientation="None"` (different from `standard_orientation=None)
    # ensures the molecule doesnt get translated or rotated at all.
    # These optimization_keywords values are the only supported ones for non-QM runs.
    # Setting the `basis="STO-2G"` reduces memory requirements for non-QM runs.
    run = exess.optimization(
        test_data_dir / "benzene_t.json",
        max_iters=100,
        optimization_keywords=exess.OptimizationKeywords(
            coordinate_system="Cartesian",
            algorithm="LBFGS",
            lbfgs_keywords=exess.LBFGSKeywords(),
        ),
        basis="STO-2G",
        standard_orientation="None",
        # ML trained on ligands, so best used for ligand fragments.
        qm_fragments=[],
        mm_fragments=[],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Optimization 01: ML",
            tags=["rush-py", "test", "benzene", "ML"],
        ),
    )
    assert_run_collects_and_caches(run, exess.OptimizationResultRef)

    result = run.fetch()
    assert isinstance(result, exess.OptimizationResult)
    assert result.trajectory
    assert result.steps

    saved = run.save()
    assert isinstance(saved, exess.OptimizationResultPaths)
    assert isinstance(saved.trajectory, Path)
    assert isinstance(saved.steps, Path)
    assert saved.trajectory.exists()
    assert saved.steps.exists()
