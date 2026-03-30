from pathlib import Path

from rush import exess
from rush import RunOpts, RunSpec
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_exports(test_data_dir: Path):
    run = exess.energy(
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
    ref = assert_run_collects_and_caches(run, exess.ResultRef)
    assert ref.exports is not None

    result = run.fetch()
    assert isinstance(result, exess.Result)
    assert isinstance(result.exports, dict)
    assert "density_descriptors" in result.exports
    assert "esp_descriptors" in result.exports

    saved = run.save()
    assert isinstance(saved, exess.ResultPaths)
    assert saved.exports is not None
    assert saved.exports.suffix == ".json"
    assert saved.calc.exists()
    assert saved.exports.exists()
