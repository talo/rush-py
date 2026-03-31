"""
Example: Hyper Solvation

This script demonstrates how to:
1. Submit a Hyper solvation run for one TRC input
2. Fetch structured output with per-item error handling
3. Save the solvated structure to the Rush workspace

Tutorial: https://exess.qdx.co/docs/tutorials/08-hyper-solvate.html

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: valid_trc.json (provided in data/)
"""

from pathlib import Path

from rush import TRC, hyper
from rush import RunOpts

DATA_DIR = Path(__file__).parent / "data"
INPUT_TRC = DATA_DIR / "valid_trc.json"

# ===== Submit Hyper solvation run =====
print("=" * 60)
print("Hyper Solvation")
print("=" * 60)

run = hyper.hyper_solvate_sumo(
    [INPUT_TRC],
    config=hyper.HyperConfig(max_inputs=8, padding_nm=0.8, seed=12345, timeout_seconds=120),
    run_opts=RunOpts(
        name="Tutorial: Hyper Solvate",
        tags=["rush-py", "tutorial", "hyper", "solvate"],
    ),
)

result_ref = run.collect()

# ===== Fetch parsed output =====
fetched = result_ref.fetch()
if len(fetched) != 1:
    raise RuntimeError(f"Expected 1 output item, got {len(fetched)}")

item = fetched[0]
if isinstance(item, hyper.ItemError):
    raise RuntimeError(f"Hyper returned per-item error: {item}")
if not isinstance(item, TRC):
    raise TypeError(f"Expected TRC output, got {type(item).__name__}")

# ===== Print output summary =====
print()
print("Results:")
print("-" * 60)
print(f"Input file: {INPUT_TRC.name}")
print(f"Solvated atom count: {len(item.topology.symbols)}")
print(f"Residue count: {len(item.residues.residues)}")

# ===== Save output artifact =====
saved = result_ref.save()
if len(saved) != 1:
    raise RuntimeError(f"Expected 1 saved item, got {len(saved)}")

saved_item = saved[0]
if isinstance(saved_item, hyper.ItemError):
    raise RuntimeError(f"Hyper returned per-item error while saving: {saved_item}")

print(f"Saved output path: {saved_item}")
print("-" * 60)
print("Done!")
