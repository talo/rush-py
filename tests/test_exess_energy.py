import json
import sys
from pathlib import Path

from rush import exess
from rush.client import RunError, RunOpts, collect_run, set_opts


def test_exess_energy_tutorial():
    res = exess.energy(
        "tests/data/6a5j_t.json",
        method="RestrictedHF",
        basis="PCSeg-0",
        ksdft_keywords=None,
        collect=True,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 00: Tutorial",
            tags=["rush-py", "test", "6a5j"],
        ),
    )
    output = exess.save_energy_outputs(res)
    assert not isinstance(output, RunError)
    output_file = output[0] if isinstance(output, tuple) else output
    with open(output_file) as f:
        print(json.load(f)["qmmbe"]["expanded_hf_energy"])


def test_exess_energy_exports():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    # Default method is RestrictedKSDFT, and default basis is cc-pVDZ
    # Using PCSeg-0 for faster test runtimes
    id = exess.energy(
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
    res = collect_run(id)
    print(res, file=sys.stderr)

    # Each module has a `save_outputs` function that automatically writes the
    # outputs as files to the workspace dir
    exess.save_energy_outputs(res)


if __name__ == "__main__":
    test_exess_energy_tutorial()
    test_exess_energy_exports()
