"""Example: Hyper solvation workflow."""

from pathlib import Path

from rush import RunOpts, TRC, hyper

DATA_DIR = Path(__file__).parent / "data"
INPUT_TRC = DATA_DIR / "valid_trc.json"

run = hyper.hyper_solvate_sumo(
    [INPUT_TRC],
    config=hyper.HyperConfig(max_inputs=8, padding_nm=0.8, seed=12345, timeout_seconds=120),
    run_opts=RunOpts(name="Example: Hyper Solvate", tags=["rush-py", "example", "hyper", "solvate"]),
)

result_ref = run.collect()
fetched = result_ref.fetch()
if len(fetched) != 1:
    raise RuntimeError(f"Expected 1 output item, got {len(fetched)}")

item = fetched[0]
if not isinstance(item, TRC):
    raise RuntimeError(f"Expected TRC output, got {item}")

print("Solvated atom count:", len(item.topology.symbols))

saved = result_ref.save()
if len(saved) != 1:
    raise RuntimeError(f"Expected 1 saved item, got {len(saved)}")

saved_item = saved[0]
if isinstance(saved_item, hyper.ItemError):
    raise RuntimeError(f"Unexpected per-item save error: {saved_item}")

print("Saved output:", saved_item)
