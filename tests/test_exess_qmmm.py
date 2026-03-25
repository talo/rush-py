from pathlib import Path

from rush import exess
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_qmmm(test_data_dir: Path):
    run = exess.qmmm(
        (test_data_dir / "6a5j_t.json", test_data_dir / "6a5j_r.json"),
        n_timesteps=500,
        temperature_kelvin=300.0,
        method="RestrictedHF",
        ksdft_keywords=None,
        # TODO: make this work (currently having convergence issues)
        # restraints=exess_qmmm.Restraints(free_fragments=[6]),
        qm_fragments=[6],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS QMMM 01: QM+MM",
            tags=["rush-py", "test", "6a5j"],
        ),
    )
    assert_run_collects_and_caches(run, exess.QMMMResultRef)

    result = run.fetch()
    assert isinstance(result, exess.QMMMResult)
    assert result.geometries
    assert all(geometry for geometry in result.geometries)

    saved = run.save()
    assert isinstance(saved, exess.QMMMResultPaths)
    assert saved.output.suffix == ".json"
    assert saved.output.exists()
