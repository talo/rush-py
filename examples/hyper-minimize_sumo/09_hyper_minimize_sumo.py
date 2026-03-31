"""Example: Hyper minimization workflow."""

from pathlib import Path

from rush import RunOpts, TRC, hyper

DATA_DIR = Path(__file__).parent / "data"

run = hyper.hyper_minimize_sumo(
    [
        hyper.MinimizeInput(
            structure=DATA_DIR / "methanol_trc.json",
            topology=DATA_DIR / "methanol_topology.json",
        )
    ],
    config=hyper.HyperMinimizeConfig(max_inputs=4, steps=100, gtol=100.0, timeout_seconds=900),
    run_opts=RunOpts(name="Example: Hyper Minimize", tags=["rush-py", "example", "hyper", "minimize"]),
)

result_ref = run.collect()
fetched = result_ref.fetch()
if len(fetched) != 1:
    raise RuntimeError(f"Expected 1 output item, got {len(fetched)}")

item = fetched[0]
if not isinstance(item, TRC):
    raise RuntimeError(f"Expected TRC output, got {item}")

print("Minimized atom count:", len(item.topology.symbols))

saved = result_ref.save()
if len(saved) != 1:
    raise RuntimeError(f"Expected 1 saved item, got {len(saved)}")

saved_item = saved[0]
if isinstance(saved_item, hyper.ItemError):
    raise RuntimeError(f"Unexpected per-item save error: {saved_item}")

print("Saved output:", saved_item)
