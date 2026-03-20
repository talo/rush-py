import sys
from pathlib import Path

import pytest

from rush import exess_geo_opt
from rush.client import RunOpts, set_opts
from rush.exess_geo_opt import (
    ExessGeoOptResult,
    ExessGeoOptSavedResult,
    fetch_outputs,
    save_outputs,
)
from rush.exess_geo_opt import exess_geo_opt as run_exess_geo_opt


@pytest.mark.skip(reason="ML regions are disabled upstream for now.")
def test_exess_optimization():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    # Setting `standard_orientation="None"` (different from `standard_orientation=None)
    # ensures the molecule doesnt get translated or rotated at all.
    # These optimization_keywords values are the only supported ones for non-QM runs.
    # Setting the `basis="STO-2G"` reduces memory requirements for non-QM runs.
    res = run_exess_geo_opt(
        max_iters=100,
        topology_path=data_dir / "benzene_t.json",
        optimization_keywords=exess_geo_opt.OptimizationKeywords(
            coordinate_system="Cartesian",
            algorithm="LBFGS",
            lbfgs_keywords=exess_geo_opt.LBFGSKeywords(),
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
        collect=True,
    )
    print(res, file=sys.stderr)
    fetched = fetch_outputs(res)
    assert isinstance(fetched, ExessGeoOptResult)
    assert fetched.trajectory
    assert fetched.steps

    saved = save_outputs(res)
    assert isinstance(saved, ExessGeoOptSavedResult)
    assert isinstance(saved.trajectory, Path)
    assert isinstance(saved.steps, Path)


if __name__ == "__main__":
    test_exess_optimization()
