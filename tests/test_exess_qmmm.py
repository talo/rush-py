import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts


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
    print(run, file=sys.stderr)
    fetched = run.fetch()
    assert isinstance(fetched, exess.QMMMResult)
    assert fetched.geometries
    print(run.save(), file=sys.stderr)
