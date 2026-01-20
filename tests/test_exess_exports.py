import sys
from pathlib import Path
from pprint import pp

from rush import exess
from rush.client import RunOpts, RunSpec, save_json, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = exess.energy(
        data_dir / "benzene_t.json",
        frag_keywords=None,  # No fragmentation; whole system calc
        export_keywords=exess.ExportKeywords(
            export_density_descriptors=True,
            export_esp_descriptors=True,
            descriptor_grid=exess.RegularDescriptorGrid(
                min=[0.0, 0.0, 0.0],
                max=[1.9, 2.0, 2.1],
                spacing=[1.0, 1.0, 1.0],
            ),
        ),
        convert_hdf5_to_json=True,
        run_spec=RunSpec(storage=1000, gpus=1),
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 04: Electron Density and ESP",
            tags=["rush-py", "test", "1kuw", "electron density", "ESP"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    files = exess.save_energy_outputs(res, to_json=True)
    print(files, file=sys.stderr)
