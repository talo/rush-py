import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts
from rush.exess import energy


def test_exess_energy_dft_hyb(test_data_dir: Path):
    run = energy(
        test_data_dir / "benzene_t.json",
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
    result = run.collect()
    print(result, file=sys.stderr)
    result.save()


def test_exess_energy_dft_dhyb(test_data_dir: Path):
    run = energy(
        test_data_dir / "benzene_t.json",
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
    result = run.collect()
    print(result, file=sys.stderr)
    result.save()
