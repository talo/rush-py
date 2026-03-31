import json
from pathlib import Path

from rush import TRC, hyper
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_hyper_solvate_sumo(test_data_dir: Path):
    run = hyper.hyper_solvate_sumo(
        [test_data_dir / "hyper" / "valid_trc.json"],
        max_inputs=8,
        padding_nm=0.8,
        seed=12345,
        timeout_seconds=120,
        run_opts=RunOpts(name="Rush-Py Test Hyper Solvate", tags=["rush-py", "test"]),
    )
    assert_run_collects_and_caches(run, hyper.SolvateResultRef)

    outputs = run.fetch()
    assert len(outputs) == 1
    assert isinstance(outputs[0], TRC)

    saved = run.save()
    assert len(saved) == 1
    assert saved[0].suffix == ".json"
    assert saved[0].exists()


def test_hyper_minimize_sumo(test_data_dir: Path):
    input_trc = json.loads((test_data_dir / "hyper" / "methanol_trc.json").read_text())
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
        run_opts=RunOpts(name="Rush-Py Test Hyper Minimize", tags=["rush-py", "test"]),
    )
    assert_run_collects_and_caches(run, hyper.MinimizeResultRef)

    outputs = run.fetch()
    assert len(outputs) == 1
    minimized = outputs[0]
    assert isinstance(minimized, TRC)
    assert len(minimized.topology.symbols) == len(input_trc["topology"]["symbols"])

    saved = run.save()
    assert len(saved) == 1
    assert saved[0].suffix == ".json"
    assert saved[0].exists()


def test_hyper_run_sumo(test_data_dir: Path):
    run = hyper.hyper_run_sumo(
        [
            hyper.RunInput(
                sim_config=test_data_dir / "hyper" / "sim_config.json",
                topology=test_data_dir / "hyper" / "methanol_topology.json",
                coordinates=test_data_dir / "hyper" / "methanol_trc.json",
            )
        ],
        max_inputs=4,
        nsteps=20,
        dt_ps=0.001,
        temperature_k=310.0,
        ensemble="Nvt",
        minimize_before_run=False,
        solvate_before_run=False,
        use_gpu=False,
        nthreads=1,
        timeout_seconds=900,
        run_opts=RunOpts(name="Rush-Py Test Hyper Run", tags=["rush-py", "test"]),
    )
    assert_run_collects_and_caches(run, hyper.RunResultRef)

    outputs = run.fetch()
    assert len(outputs) == 1
    run_output = outputs[0]
    assert isinstance(run_output, hyper.HyperRunOutput)
    assert isinstance(run_output.trajectory, bytes)
    assert len(run_output.trajectory) > 0

    saved = run.save()
    assert len(saved) == 1
    saved_output = saved[0]
    assert isinstance(saved_output, hyper.HyperRunOutputPaths)
    assert saved_output.trajectory.suffix == ".xtc"
    assert saved_output.trajectory.exists()
    if saved_output.checkpoint is not None:
        assert saved_output.checkpoint.exists()
