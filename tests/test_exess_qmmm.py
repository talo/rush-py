import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, save_object, set_opts


def test_exess_qmmm():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = exess.qmmm(
        n_timesteps=500,
        temperature_kelvin=300.0,
        topology_path=data_dir / "6a5j_t.json",
        residues_path=data_dir / "6a5j_r.json",
        method="RestrictedHF",
        ksdft_keywords=None,
        # TODO: make this work (currently having convergence issues)
        # restraints=exess.Restraints(free_fragments=[6]),
        qm_fragments=[6],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS QMMM 01: QM+MM",
            tags=["rush-py", "test", "6a5j"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    # Single-output results get returned direclty, not inside a list or tuple
    save_object(res["path"])


if __name__ == "__main__":
    test_exess_qmmm()
