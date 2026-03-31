"""Example: Hyper molecular dynamics run workflow."""

from pathlib import Path

from rush import RunOpts, hyper

DATA_DIR = Path(__file__).parent / "data"

run = hyper.hyper_run_sumo(
    [
        hyper.RunInput(
            sim_config=DATA_DIR / "sim_config.json",
            topology=DATA_DIR / "methanol_topology.json",
            coordinates=DATA_DIR / "methanol_trc.json",
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
        timeout_seconds=900,
    ),
    run_opts=RunOpts(name="Example: Hyper Run", tags=["rush-py", "example", "hyper", "run"]),
)

result_ref = run.collect()
fetched = result_ref.fetch()
if len(fetched) != 1:
    raise RuntimeError(f"Expected 1 output item, got {len(fetched)}")

item = fetched[0]
if isinstance(item, hyper.ItemError):
    raise RuntimeError(f"Unexpected per-item run error: {item}")

print("Trajectory bytes:", len(item.trajectory))
print("Checkpoint bytes:", 0 if item.checkpoint is None else len(item.checkpoint))

saved = result_ref.save()
if len(saved) != 1:
    raise RuntimeError(f"Expected 1 saved item, got {len(saved)}")

saved_item = saved[0]
if isinstance(saved_item, hyper.ItemError):
    raise RuntimeError(f"Unexpected per-item save error: {saved_item}")

print("Saved trajectory:", saved_item.trajectory)
print("Saved checkpoint:", saved_item.checkpoint)
