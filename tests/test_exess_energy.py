import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, set_opts
from rush.exess import energy


def test_exess_energy_tutorial():
    run = energy(
        "tests/data/6a5j_t.json",
        method="RestrictedHF",
        basis="PCSeg-0",
        ksdft_keywords=None,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 00: Tutorial",
            tags=["rush-py", "test", "6a5j"],
        ),
    )
    output = run.fetch()
    assert output.calc.qmmbe is not None
    assert output.calc.qmmbe.reference_fragment is None
    assert output.calc.qmmbe.expanded_hf_energy is not None
    print(output.calc.qmmbe.expanded_hf_energy)


def test_exess_energy_exports():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    # Default method is RestrictedKSDFT, and default basis is cc-pVDZ
    # Using PCSeg-0 for faster test runtimes
    run = energy(
        data_dir / "6a5j_t.json",
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
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 01: w/ Export Keywords",
            tags=["rush-py", "test", "6a5j"],
        ),
    )
    result = run.collect()
    print(result, file=sys.stderr)

    # Each module result has .save() for downloading outputs to the workspace
    result.save()


if __name__ == "__main__":
    test_exess_energy_tutorial()
    test_exess_energy_exports()
