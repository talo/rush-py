import time
from pathlib import Path

from rush.client import (
    RunOpts,
    delete_run,
    fetch_runs,
)
from rush.exess import energy


def test_delete_run(test_data_dir: Path):
    run = energy(
        test_data_dir / "1kuw_t.json",
        basis="PCSeg-0",
        run_opts=RunOpts(
            name="Rush-Py Test Delete Run 01",
            tags=["1kuw", "delete-me"],
        ),
    )
    runs_0 = fetch_runs(tags=["delete-me"])
    assert run.id in runs_0
    delete_run(run.id)
    time.sleep(1)
    runs_1 = fetch_runs(tags=["delete-me"])
    assert run.id not in runs_1
