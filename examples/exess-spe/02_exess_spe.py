"""
Example: EXESS Single Point Energy Calculation

This script demonstrates how to:
1. Run a single-point energy (SPE) calculation using Rush
2. Extract and print the total energy and electronic properties

Tutorial: https://exess.qdx.co/docs/tutorials/02-exess-spe.html

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: water_topology.json (provided in data/)
"""

from pathlib import Path

from rush import exess
from rush.client import RunOpts
from rush.exess import exess_energy

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

outputs = exess_energy(
    TOPOLOGY_FILE,
    method=METHOD,
    basis=BASIS,
    run_opts=RunOpts(
        name="Rush-Py Tutorial: Single Point Energy",
        tags=["rush-py", "tutorial", "exess", "spe"],
    ),
    collect=True,
)

# Parse fetched outputs in memory
res = exess.fetch_outputs(outputs)

# Access total energy from qmmbe object
total_energy = res.calc.qmmbe.expanded_hf_energy

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
