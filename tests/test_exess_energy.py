from pathlib import Path

from rush import exess
from rush import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_energy_tutorial(test_data_dir: Path):
    run = exess.energy(
        test_data_dir / "6a5j_t.json",
        method="RestrictedHF",
        basis="PCSeg-0",
        ksdft_keywords=None,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 00: Tutorial",
            tags=["rush-py", "test", "6a5j"],
        ),
    )
    assert_run_collects_and_caches(run, exess.ResultRef)

    result = run.fetch()
    assert isinstance(result, exess.Result)
    assert result.exports == {}
    assert result.calc.qmmbe.reference_fragment is None
    assert result.calc.qmmbe.expanded_hf_energy is not None

    saved = run.save()
    assert isinstance(saved, exess.ResultPaths)
    assert saved.calc.suffix == ".json"
    assert saved.calc.exists()


def test_exess_energy_exports(test_data_dir: Path):
    # Default method is RestrictedKSDFT, and default basis is cc-pVDZ
    # Using PCSeg-0 for faster test runtimes
    run = exess.energy(
        test_data_dir / "6a5j_t.json",
        method="RestrictedHF",
        basis="PCSeg-0",
        ksdft_keywords=None,
        system=exess.System(max_gpu_memory_mb=5000),
        export_keywords=exess.ExportKeywords(
            export_density=True,
            export_molecular_orbital_coeffs=True,
            export_mulliken_charges=True,
            export_chelpg_charges=True,
            export_basis_labels=True,
        ),
        convert_hdf5_to_json=False,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 01: w/ Export Keywords",
            tags=["rush-py", "test", "6a5j"],
        ),
    )
    ref = assert_run_collects_and_caches(run, exess.ResultRef)
    assert ref.exports is not None

    result = run.fetch()
    assert isinstance(result, exess.Result)
    assert isinstance(result.exports, bytes)
    assert result.exports

    saved = run.save()
    assert isinstance(saved, exess.ResultPaths)
    assert saved.exports is not None
    assert saved.exports.suffix == ".hdf5"
    assert saved.calc.exists()
    assert saved.exports.exists()
