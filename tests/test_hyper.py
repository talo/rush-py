from __future__ import annotations

from pathlib import Path

from rush import TRC, hyper
from rush import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches



def test_hyper_solvate_sumo(test_data_dir: Path):
    run = hyper.hyper_solvate_sumo(
        [test_data_dir / "hyper" / "valid_trc.json"],
        max_inputs=8,
        padding_nm=0.8,
        seed=12345,
        timeout_seconds=120,
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Solvate 01",
            tags=["rush-py", "test", "hyper", "solvate"],
        ),
    )
    assert_run_collects_and_caches(run, hyper.SolvateResultRef)

    fetched = run.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], TRC)

    saved = run.save()
    assert len(saved) == 1
    assert isinstance(saved[0], Path)
    assert saved[0].exists()


def test_hyper_minimize_sumo(test_data_dir: Path):
    run = hyper.hyper_minimize_sumo(
        [
            hyper.MinimizeInput(
                structure=test_data_dir / "hyper" / "methanol_trc.json",
                topology=test_data_dir / "hyper" / "methanol_topology.json",
            )
        ],
        max_inputs=4,
        steps=100,
        gtol=100.0,
        timeout_seconds=900,
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Minimize 01",
            tags=["rush-py", "test", "hyper", "minimize"],
        ),
    )
    assert_run_collects_and_caches(run, hyper.MinimizeResultRef)

    fetched = run.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], TRC)

    saved = run.save()
    assert len(saved) == 1
    assert isinstance(saved[0], Path)
    assert saved[0].exists()


def test_hyper_run_sumo(test_data_dir: Path):
    run = hyper.hyper_run_sumo(
        [
            hyper.RunInput(
                sim_config_json=test_data_dir / "hyper" / "sim_config_string.json",
                topology=test_data_dir / "hyper" / "methanol_topology.json",
                coordinates=test_data_dir / "hyper" / "methanol_trc.json",
            )
        ],
        max_inputs=8,
        timeout_seconds=1800,
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Run 01",
            tags=["rush-py", "test", "hyper", "run"],
        ),
    )
    assert_run_collects_and_caches(run, hyper.RunResultRef)

    fetched = run.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], hyper.RunOutput)
    assert len(fetched[0].trajectory) > 0

    saved = run.save()
    assert len(saved) == 1
    assert isinstance(saved[0], hyper.RunOutputPaths)
    assert saved[0].trajectory.exists()
    assert saved[0].trajectory.suffix == ".xtc"
    if saved[0].checkpoint is not None:
        assert saved[0].checkpoint.exists()
        assert saved[0].checkpoint.suffix == ".bin"
