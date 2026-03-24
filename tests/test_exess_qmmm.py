import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, set_opts


def test_exess_qmmm():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    run = exess.qmmm(
        (data_dir / "6a5j_t.json", data_dir / "6a5j_r.json"),
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


if __name__ == "__main__":
    test_exess_qmmm()
