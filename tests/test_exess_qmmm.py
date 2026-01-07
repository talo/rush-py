import sys
from pathlib import Path

from rush import exess
from rush.client import RunOpts, save_object, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = exess.qmmm(
        data_dir / "6a5j_t.json",
        data_dir / "6a5j_r.json",
        n_timesteps=500,
        # TODO: make this work (currently having convergence issues)
        # restraints=exess.Restraints(free_fragments=[6]),
        qm_fragments=[6],
        ml_fragments=[],
        run_opts=RunOpts(
            name="Rush-Py Test EXESS QMMM 01: QM+MM",
            tags=["rush-py", "test", "6a5j"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    # Single-output results get returned direclty, not inside a list or tuple
    save_object(res["path"])
