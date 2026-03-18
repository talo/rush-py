import sys
from pathlib import Path

from rush.client import RunError, RunOpts, set_opts
from rush.exess_qmmm import ExessQMMMResult, fetch_outputs, save_outputs
from rush.exess_qmmm import exess_qmmm as run_exess_qmmm


def test_exess_qmmm():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = run_exess_qmmm(
        n_timesteps=500,
        temperature_kelvin=300.0,
        topology_path=data_dir / "6a5j_t.json",
        residues_path=data_dir / "6a5j_r.json",
        method="RestrictedHF",
        ksdft_keywords=None,
        # TODO: make this work (currently having convergence issues)
        # restraints=exess_qmmm.Restraints(free_fragments=[6]),
        qm_fragments=[6],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS QMMM 01: QM+MM",
            tags=["rush-py", "test", "6a5j"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    fetched = fetch_outputs(res)
    assert isinstance(fetched, ExessQMMMResult)
    assert fetched.geometries

    saved = save_outputs(res)
    assert isinstance(saved, Path)

    assert not isinstance(fetched, RunError)


if __name__ == "__main__":
    test_exess_qmmm()
