import sys
from pathlib import Path

from rush_py2 import exess
from rush_py2.client import RunOpts, save_object, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = exess.optimization(
        max_iters=100,
        topology_path=data_dir / "6a5j_t.json",
        residues_path=data_dir / "6a5j_r.json",
        optimization_keywords=exess.OptimizationKeywords(
            coordinate_system="Cartesian",
            algorithm="LBFGS",
            lbfgs_keywords=exess.LBFGSKeywords(),
        ),
        standard_orientation="None",
        qm_fragments=[0],
        ml_fragments=[],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Optimization 04: QM+MM",
            tags=["rush-py", "test", "6a5j", "QM+MM"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    for res_i in res:
        save_object(res_i["path"])
