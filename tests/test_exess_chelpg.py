from pathlib import Path

from rush import exess, from_json
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_energy_chelpg_1hsg_MK1(test_data_dir: Path):
    trc = from_json(test_data_dir / "1hsg_MK1_trc.json")[0]

    run = exess.energy(
        trc,
        basis="PCSeg-0",
        frag_keywords=None,  # Important, to disable fragmentation
        export_keywords=exess.ExportKeywords(export_chelpg_charges=True),
        convert_hdf5_to_json=True,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 03.1: ChelpG via Energy",
            tags=["rush-py", "test", "tyk2+ejm-31"],
        ),
    )
    ref = assert_run_collects_and_caches(run, exess.ResultRef)
    assert ref.exports is not None

    result = run.fetch()
    assert isinstance(result.exports, dict)
    charges = result.exports["chelpg_charges"]
    assert charges

    saved = run.save()
    assert isinstance(saved, exess.ResultPaths)
    assert saved.exports is not None
    assert saved.exports.suffix == ".json"
    assert saved.calc.exists()
    assert saved.exports.exists()


def test_exess_energy_chelpg_benzene(test_data_dir: Path):
    run = exess.energy(
        test_data_dir / "benzene_t.json",
        method="RestrictedRIMP2",
        basis="def2-TZVP",
        aux_basis="def2-TZVP-RIFIT",
        scf_keywords=exess.SCFKeywords(fock_build_type="RI"),
        frag_keywords=None,  # Important, to disable fragmentation
        export_keywords=exess.ExportKeywords(export_chelpg_charges=True),
        convert_hdf5_to_json=True,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 03.2: ChelpG via Energy",
            tags=["rush-py", "test", "benzene"],
        ),
    )
    ref = assert_run_collects_and_caches(run, exess.ResultRef)
    assert ref.exports is not None

    result = run.fetch()
    assert isinstance(result.exports, dict)
    charges = result.exports["chelpg_charges"]
    assert charges

    saved = run.save()
    assert isinstance(saved, exess.ResultPaths)
    assert saved.exports is not None
    assert saved.exports.suffix == ".json"
    assert saved.calc.exists()
    assert saved.exports.exists()
