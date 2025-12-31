import sys
from pathlib import Path
from pprint import pp

from rush_py2 import exess
from rush_py2.client import RunOpts, RunSpec, save_json, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = exess.energy(
        data_dir / "1kuw_t.json",
        export_keywords=exess.ExportKeywords(
            export_density_descriptors=True,
            export_esp_descriptors=True,
            descriptor_grid=exess.CustomDescriptorGrid(
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            ),
        ),
        run_spec=RunSpec(storage=1000, gpus=1),
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 04: Electron Density and ESP",
            tags=["rush-py", "test", "1kuw", "electron density", "ESP"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    exess.save_energy_outputs(res)
