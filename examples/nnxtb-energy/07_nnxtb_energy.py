"""
Example: NN-xTB Energy and Forces

This script demonstrates how to:
1. Run an NN-xTB energy and forces calculation using Rush
2. Fetch the parsed result with `fetch_outputs`
3. Inspect per-atom forces

Tutorial: https://exess.qdx.co/docs/tutorials/07-nnxtb-energy.html

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: 1kuw_t.json (provided in data/)
"""
from pathlib import Path

from rush.client import RunError, RunOpts
from rush.nnxtb import fetch_outputs, nnxtb

DATA_DIR = Path(__file__).parent / "data"
TOPOLOGY_FILE = DATA_DIR / "1kuw_t.json"

# ===== Run NN-xTB energy and forces calculation =====
print("=" * 60)
print("NN-xTB Energy and Forces Calculation")
print("=" * 60)

res = nnxtb(
    TOPOLOGY_FILE,
    compute_forces=True,
    run_opts=RunOpts(
        name="Tutorial: NN-xTB Energy",
        tags=["rush-py", "tutorial", "nnxtb"],
    ),
    collect=True,
)

if isinstance(res, RunError):
    print(f"Run failed: {res.message}")
    exit(1)

# ===== Parse results =====
results = fetch_outputs(res)

if isinstance(results, RunError):
    print(f"Output fetch failed: {results.message}")
    exit(1)

# ===== Print energy =====
print()
print("Results:")
print("-" * 40)
print(f"  Energy:  {results.energy_mev:.2f} meV")
print(f"           {results.energy_mev / 1000:.6f} eV")
print(f"           {results.energy_mev / 1000 * 23.06:.4f} kcal/mol")

# ===== Print forces =====
if results.forces_mev_per_angstrom:
    print()
    print(f"  Forces ({len(results.forces_mev_per_angstrom)} atoms):")
    print(f"  {'Atom':<6} {'Fx':>10} {'Fy':>10} {'Fz':>10} {'|F|':>10}  (meV/A)")
    for i, (fx, fy, fz) in enumerate(results.forces_mev_per_angstrom):
        magnitude = (fx**2 + fy**2 + fz**2) ** 0.5
        print(f"  {i:<6} {fx:>10.2f} {fy:>10.2f} {fz:>10.2f} {magnitude:>10.2f}")
        if i >= 9:
            remaining = len(results.forces_mev_per_angstrom) - 10
            if remaining > 0:
                print(f"  ... ({remaining} more atoms)")
            break

print("-" * 40)
print("\nDone!")
