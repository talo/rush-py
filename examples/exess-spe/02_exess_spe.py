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

# Save outputs
files = exess.save_energy_outputs(res)
print(f"Saved files: {files}")

# Extract energy from JSON output
total_energy = None
for output in res:
    if isinstance(output, dict):
        if 'path' in output and output.get("format") == "json":
            # Flat dict format with embedded data
            json_data = output.get("data", {})
            if "total_energy" in json_data:
                total_energy = json_data["total_energy"]
                break
        elif "Json" in output:
            # Type-discriminated format
            json_data = output["Json"].get("data", {})
            if "total_energy" in json_data:
                total_energy = json_data["total_energy"]
                break

# Fallback: try loading from saved JSON file
if total_energy is None:
    for f in files:
        if str(f).endswith(".json"):
            try:
                with open(f) as fh:
                    energy_data = json.load(fh)
                    # Check for total_energy at top level
                    if "total_energy" in energy_data:
                        total_energy = energy_data["total_energy"]
                    # Or check for expanded_hf_energy in qmmbe object
                    elif "qmmbe" in energy_data and "expanded_hf_energy" in energy_data["qmmbe"]:
                        total_energy = energy_data["qmmbe"]["expanded_hf_energy"]
                    break
            except (json.JSONDecodeError, IOError):
                pass

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
