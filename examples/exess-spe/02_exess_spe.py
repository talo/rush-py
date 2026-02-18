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
from rush.client import RunOpts

DATA_DIR = Path(__file__).parent / "data"
TOPOLOGY_FILE = DATA_DIR / "water_topology.json"

METHOD = "RestrictedHF"
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

# Save outputs
files = exess.save_energy_outputs(res)
print(f"Saved files: {files}")

# Extract energy from JSON output
energy_data = None
for output in res:
    if isinstance(output, dict):
        if "Json" in output:
            json_path = output["Json"]["path"]
            # The data may be embedded or need downloading
            energy_data = output["Json"].get("data", {})
            break
        elif output.get("format") == "json":
            energy_data = output.get("data", {})
            break

# Also try loading from saved JSON file
if not energy_data or "total_energy" not in energy_data:
    for f in files:
        if str(f).endswith(".json"):
            with open(f) as fh:
                energy_data = json.load(fh)
            break

total_energy = energy_data.get("total_energy") if energy_data else None
dipole = energy_data.get("dipole_moment") if energy_data else None

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

if dipole is not None:
    if isinstance(dipole, list) and len(dipole) == 3:
        mag = (dipole[0]**2 + dipole[1]**2 + dipole[2]**2) ** 0.5
        print(f"  Dipole Moment: [{dipole[0]:.6f}, {dipole[1]:.6f}, {dipole[2]:.6f}]")
        print(f"                 |μ| = {mag:.6f} a.u.")
    else:
        print(f"  Dipole Moment: {dipole}")

print("-" * 40)
print("\n✓ All done!")
