import sys
from pathlib import Path

from rush import exess, exess_geo_opt
from rush.client import RunOpts, save_object, set_opts
from rush.exess_geo_opt import exess_geo_opt as run_exess_geo_opt


def test_exess_optimization_qm_mm():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = run_exess_geo_opt(
        max_iters=100,
        optimization_keywords=exess_geo_opt.OptimizationKeywords(
            coordinate_system="Cartesian",
            algorithm="LBFGS",
            lbfgs_keywords=exess_geo_opt.LBFGSKeywords(),
        ),
        topology_path=data_dir / "6a5j_t.json",
        residues_path=data_dir / "6a5j_r.json",
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
        collect=True,
    )
    print(res, file=sys.stderr)
    for res_i in res:
        save_object(res_i["path"])


if __name__ == "__main__":
    test_exess_optimization_qm_mm()
