import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, collect_run, set_opts


def test_exess_energy_dft_hyb():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path(__file__).parent / "data"
    id = exess.energy(
        data_dir / "benzene_t.json",
        method="RestrictedKSDFT",
        ksdft_keywords=exess.KSDFTKeywords(
            functional="HYB_GGA_XC_B3LYP",
            grid=exess.XCGridParameters(
                radial_quad="TreutlerAldrichs",
                pruning_scheme="Treutler",
                resolution=exess.DefaultGridResolution("TreutlerGM5"),
                batching=exess.SpaceFillingBatching(),
            ),
            method="BatchDense",
            sp_threshold=1e-12,
            dp_threshold=1e-11,
            batches_per_batch=10,
        ),
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 05: DFT (Hybrid)",
            tags=["rush-py", "test", "dft", "HYB_GGA_XC_B3LYP", "benzene"],
        ),
    )
    res = collect_run(id)
    print(res, file=sys.stderr)

    # Each module has a `save_outputs` function that automatically writes the
    # outputs as files to the workspace dir
    exess.save_outputs(res)


def test_exess_energy_dft_dhyb():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path(__file__).parent / "data"
    id = exess.energy(
        data_dir / "benzene_t.json",
        method="RestrictedKSDFT",
        basis="cc-pVTZ",
        aux_basis="cc-pVTZ-RIFIT",
        ksdft_keywords=exess.KSDFTKeywords(
            functional="revDSD-PBEP86-D4",
            grid=exess.XCGridParameters(batching=exess.SpaceFillingBatching()),
        ),
        scf_keywords=exess.SCFKeywords(fock_build_type="RI"),
        frag_keywords=None,  # must disable frag for double hybrid
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 06: DFT (Double Hybrid)",
            tags=["rush-py", "test", "dft", "revDSD-PBEP86-D4", "benzene"],
        ),
    )
    res = collect_run(id)
    print(res, file=sys.stderr)

    # Each module has a `save_outputs` function that automatically writes the
    # outputs as files to the workspace dir
    exess.save_outputs(res)


if __name__ == "__main__":
    test_exess_energy_dft_hyb()
    test_exess_energy_dft_dhyb()
