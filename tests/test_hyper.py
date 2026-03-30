from pathlib import Path

from rush import TRC, hyper
from rush.client import RunOpts, RunSpec
from tests._module_test_utils import assert_run_collects_and_caches


def test_hyper_solvate_sumo(test_data_dir: Path):
    run = hyper.hyper_solvate_sumo(
        [test_data_dir / "hyper" / "valid_trc.json"],
        config=hyper.HyperConfig(
            max_inputs=8,
            padding_nm=0.8,
            seed=12345,
            timeout_seconds=120,
        ),
        run_spec=RunSpec(storage=4096),
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Solvate 01",
            tags=["rush-py", "test", "hyper", "solvate"],
        ),
    )
    ref = assert_run_collects_and_caches(run, hyper.SolvateResultRef)
    assert len(ref) == 1

    result = run.fetch()
    assert len(result) == 1
    assert isinstance(result[0], TRC)

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
        config=hyper.HyperMinimizeConfig(
            max_inputs=4,
            steps=100,
            gtol=100.0,
            timeout_seconds=900,
        ),
        run_spec=RunSpec(storage=4096),
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Minimize 01",
            tags=["rush-py", "test", "hyper", "minimize"],
        ),
    )
    ref = assert_run_collects_and_caches(run, hyper.MinimizeResultRef)
    assert len(ref) == 1

    result = run.fetch()
    assert len(result) == 1
    assert isinstance(result[0], TRC)

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
        config=hyper.HyperRunConfig(
            max_inputs=8,
            timeout_seconds=1800,
        ),
        run_spec=RunSpec(storage=4096),
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Run 01",
            tags=["rush-py", "test", "hyper", "run"],
        ),
    )
    ref = assert_run_collects_and_caches(run, hyper.RunResultRef)
    assert len(ref) == 1

    result = run.fetch()
    assert len(result) == 1
    assert isinstance(result[0], hyper.RunOutput)
    assert len(result[0].trajectory) > 0

    saved = run.save()
    assert len(saved) == 1
    assert isinstance(saved[0], hyper.RunOutputPaths)
    assert saved[0].trajectory.exists()
    if saved[0].checkpoint is not None:
        assert saved[0].checkpoint.exists()
