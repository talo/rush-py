import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, RunSpec
from rush.exess import energy


def test_exess_exports(test_data_dir: Path):
    run = energy(
        test_data_dir / "benzene_t.json",
        basis="PCSeg-0",
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
    )
    result = run.collect()
    print(result, file=sys.stderr)
    files = result.save()
    print(files, file=sys.stderr)
