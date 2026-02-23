"""
Example: EXESS Single Point Energy Calculation

This script demonstrates how to:
1. Run a single-point energy (SPE) calculation using Rush
2. Extract and print the total energy and electronic properties

Tutorial: docs/tutorials/02-exess-spe.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: water_topology.json (provided in data/)
"""

import json
from pathlib import Path

from rush import exess
from rush.client import RunOpts, RunError

DATA_DIR = Path(__file__).parent / "data"
TOPOLOGY_FILE = DATA_DIR / "water_topology.json"

METHOD = "RestrictedHF"
# ⚠️ TUTORIAL ONLY: STO-3G is a minimal basis set used here for speed/demonstration.
# It is NOT suitable for research or production use. For real work, use at least
# cc-pVDZ or larger (e.g., cc-pVTZ, aug-cc-pVDZ) with an appropriate method.
BASIS = "STO-3G"

# ===== Run single-point energy calculation =====
print("=" * 60)
print("Single Point Energy Calculation: Water (H₂O)")
print(f"Method: {METHOD}/{BASIS}")
print("=" * 60)

res = exess.energy(
    TOPOLOGY_FILE,
    method=METHOD,
    basis=BASIS,
    run_opts=RunOpts(
        name="Rush-Py Tutorial: Single Point Energy",
        tags=["rush-py", "tutorial", "exess", "spe"],
    ),
    collect=True,
)

if isinstance(res, RunError):
    print(f"Run failed: {res.message}")
    exit(1)

# Save energy outputs to disk and load from file
files = exess.save_energy_outputs(res)
json_file = next((f for f in files if str(f).endswith('.json')), None)
with open(json_file, encoding='utf-8') as f:
    energy_data = json.load(f)

# Access total energy from qmmbe object
total_energy = energy_data.get("qmmbe", {}).get("expanded_hf_energy")

# ===== Print results =====
print()
print("Results:")
print("-" * 40)
if total_energy is not None:
    print(f"  Total Energy:  {total_energy:.10f} Hartree")
    print(f"                 {total_energy * 627.509474:.4f} kcal/mol")
    print(f"                 {total_energy * 2625.4996:.4f} kJ/mol")
else:
    print("  Total Energy:  (not available — check output files)")

print("-" * 40)
print("\n✓ All done!")
