from pathlib import Path

from rush import TRC, hyper
from rush.client import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_hyper_solvate_sumo(test_data_dir: Path):
    run = hyper.hyper_solvate_sumo(
        [test_data_dir / "hyper" / "valid_trc.json"],
        config=hyper.HyperConfig(max_inputs=8, padding_nm=0.8, seed=12345),
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Solvate",
            tags=["rush-py", "test", "hyper", "solvate"],
        ),
    )

    assert_run_collects_and_caches(run, hyper.TRCBatchResultRef)

    fetched = run.fetch()
    assert len(fetched) == 1
    assert all(not isinstance(item, hyper.ItemError) for item in fetched)
    assert isinstance(fetched[0], TRC)

    saved = run.save()
    assert len(saved) == 1
    assert all(not isinstance(item, hyper.ItemError) for item in saved)
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
        config=hyper.HyperMinimizeConfig(max_inputs=4, steps=100, gtol=100.0),
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Minimize",
            tags=["rush-py", "test", "hyper", "minimize"],
        ),
    )

    assert_run_collects_and_caches(run, hyper.TRCBatchResultRef)

    fetched = run.fetch()
    assert len(fetched) == 1
    assert all(not isinstance(item, hyper.ItemError) for item in fetched)
    assert isinstance(fetched[0], TRC)

    saved = run.save()
    assert len(saved) == 1
    assert all(not isinstance(item, hyper.ItemError) for item in saved)
    assert isinstance(saved[0], Path)
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
        config=hyper.HyperRunConfig(
            max_inputs=4,
            nsteps=20,
            dt_ps=0.001,
            temperature_k=300.0,
            ensemble="Nvt",
            minimize_before_run=False,
            solvate_before_run=False,
            use_gpu=False,
            nthreads=1,
        ),
        run_opts=RunOpts(
            name="Rush-Py Test Hyper Run",
            tags=["rush-py", "test", "hyper", "run"],
        ),
    )

    assert_run_collects_and_caches(run, hyper.RunResultRef)

    fetched = run.fetch()
    assert len(fetched) == 1
    assert all(not isinstance(item, hyper.ItemError) for item in fetched)
    assert isinstance(fetched[0], hyper.RunOutput)
    assert fetched[0].trajectory

    saved = run.save()
    assert len(saved) == 1
    assert all(not isinstance(item, hyper.ItemError) for item in saved)
    assert isinstance(saved[0], hyper.RunOutputPaths)
    assert saved[0].trajectory.exists()
    if saved[0].checkpoint is not None:
        assert saved[0].checkpoint.exists()
