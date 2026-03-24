import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts


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
    print(run, file=sys.stderr)
    fetched = run.fetch()
    assert isinstance(fetched, exess.OptimizationResult)
    # TODO: check why convergence fails here

    saved = run.save()
    assert isinstance(saved, exess.OptimizationResultPaths)
