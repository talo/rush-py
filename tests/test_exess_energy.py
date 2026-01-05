import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, collect_run, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    # Default method is ReducedHF, and default basis is cc-pVDZ
    # Due to choice of method, no need for aux_basis (and default is None)
    id = exess.energy(
        data_dir / "6a5j_t.json",
        basis="PCSeg-0",
        system=exess.System(max_gpu_memory_mb=1000),
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
