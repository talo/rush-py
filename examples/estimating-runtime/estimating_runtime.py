"""
Example: Estimating Runtime

This script demonstrates how to:
1. Run a geometry optimization
2. Retrieve run info (including EXESS logs) via fetch_run_info
3. Print timestamps and stdout to estimate walltime

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: ethene_twisted_t.json (provided in ../exess-optimization/data/)
"""

from pathlib import Path

from rush import exess, fetch_run_info
from rush.client import RunOpts

DATA_DIR = Path(__file__).parent / ".." / "exess-optimization" / "data"
INPUT_FILE = DATA_DIR / "ethene_twisted_t.json"

# ===== Run a geometry optimization =====
print("Submitting geometry optimization...")

run = exess.optimization(
    INPUT_FILE,
    100,
    method="RestrictedHF",
    basis="STO-3G",
    standard_orientation="None",
    run_opts=RunOpts(name="Example: Estimating Runtime"),
)

print(f"Run ID: {run.id}")

# Wait for the run to complete
res = run.fetch()
print(f"Optimization converged in {len(res.steps)} steps")

# ===== Fetch run info and print logs =====
info = fetch_run_info(run.id)
if info is None:
    print("Could not fetch run info.")
else:
    print()
    print(info)
    print()
    print(f"Created at: {info.created_at}")
    print(f"Updated at: {info.updated_at}")

    if info.stdout:
        print()
        print("=" * 60)
        print("EXESS logs (stdout)")
        print("=" * 60)
        print(info.stdout)
    else:
        print()
        print("No stdout logs available for this run.")
