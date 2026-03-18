import sys
from pathlib import Path

from rush import exess_geo_opt
from rush.client import RunOpts, save_object, set_opts
from rush.exess_geo_opt import exess_geo_opt as run_exess_geo_opt


def test_exess_optimization_mm():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = run_exess_geo_opt(
        max_iters=10000,
        topology_path=data_dir / "6a5j_t.json",
        # Residues are required for MM fragments
        residues_path=data_dir / "6a5j_r.json",
        optimization_keywords=exess_geo_opt.OptimizationKeywords(
            coordinate_system="Cartesian",
            algorithm="LBFGS",
            lbfgs_keywords=exess_geo_opt.LBFGSKeywords(),
        ),
        basis="STO-2G",
        standard_orientation="None",
        # MM fragments well for uncomplicated regions and runs very quickly.
        qm_fragments=[],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Optimization 02: MM",
            tags=["rush-py", "test", "6a5j", "MM"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    for res_i in res:
        save_object(res_i["path"])


if __name__ == "__main__":
    test_exess_optimization_mm()
