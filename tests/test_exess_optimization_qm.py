import sys
from pathlib import Path

from rush import exess_geo_opt
from rush.client import RunOpts, save_object, set_opts
from rush.exess_geo_opt import exess_geo_opt as run_exess_geo_opt


def test_exess_optimization_qm():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = run_exess_geo_opt(
        max_iters=100,
        topology_path=data_dir / "benzene_t.json",
        optimization_keywords=exess_geo_opt.OptimizationKeywords(),
        method="RestrictedRIMP2",
        basis="cc-pVDZ",
        aux_basis="cc-pVDZ-RIFIT",
        standard_orientation="None",
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Optimization 03: QM",
            tags=["rush-py", "test", "benzene", "QM"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    for res_i in res:
        save_object(res_i["path"])


if __name__ == "__main__":
    test_exess_optimization_qm()
