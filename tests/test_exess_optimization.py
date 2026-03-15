import sys
from pathlib import Path

import pytest

from rush import exess
from rush.client import RunOpts, save_object, set_opts


@pytest.mark.skip(reason="ML regions are disabled upstream for now.")
def test_exess_optimization():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    # Setting `standard_orientation="None"` (different from `standard_orientation=None)
    # ensures the molecule doesnt get translated or rotated at all.
    # These optimization_keywords values are the only supported ones for non-QM runs.
    # Setting the `basis="STO-2G"` reduces memory requirements for non-QM runs.
    res = exess.optimization(
        max_iters=100,
        topology_path=data_dir / "benzene_t.json",
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
        collect=True,
    )
    print(res, file=sys.stderr)
    # If there's no specific `save_outputs` function,
    # here a canonical way to save all the objects.
    for res_i in res:
        save_object(res_i["path"])


if __name__ == "__main__":
    test_exess_optimization()
