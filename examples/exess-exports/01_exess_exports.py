"""
Example: EXESS Data Exports

This script demonstrates how to:
1. Run an EXESS energy calculation with export keywords
2. Save and inspect the output files
3. Use descriptor grids for electron density and ESP values

Tutorial: docs/tutorials/exess-exports.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: benzene_t.json (provided in data/)
"""

from pathlib import Path
from rush import exess
from rush.client import RunOpts, RunSpec

DATA_DIR = Path(__file__).parent / "data"
TOPOLOGY_FILE = DATA_DIR / "input_topology.json"


# ===== Example 1: Basic export with electron density =====
print("=" * 60)
print("Example 1: Exporting electron density")
print("=" * 60)

# NOTE: Using RestrictedHF/STO-3G for demonstration purposes only.
# This is a very fast but low-accuracy method. For production results,
# use a higher-level method (e.g., RestrictedHF/cc-pVDZ or DFT).

res = exess.energy(
    TOPOLOGY_FILE,
    method="RestrictedHF",
    basis="STO-3G",
    export_keywords=exess.ExportKeywords(
        export_density=True,
    ),
    run_opts=RunOpts(
        name="Rush-Py Tutorial: EXESS Exports 1",
        tags=["rush-py", "tutorial", "exess"],
    ),
    collect=True,
)

# Inspect the outputs
print("Raw outputs:")
for i, output in enumerate(res):
    if 'path' in output:
        # First output: flat dict with path/format
        print(f"  [{i}] path={output['path']}, format={output.get('format', 'unknown')}")
    elif 'Json' in output:
        # Type-discriminated JSON output
        print(f"  [{i}] Json: path={output['Json']['path']}")
    elif 'Hdf5' in output:
        # Type-discriminated HDF5 output
        print(f"  [{i}] Hdf5: path={output['Hdf5']['path']}")
    else:
        print(f"  [{i}] Unknown output type with keys: {list(output.keys())}")

# Save outputs to disk (JSON + HDF5)
files = exess.save_energy_outputs(res)
print(f"Saved files: {files}")


# ===== Example 2: Descriptor grids for density and ESP =====
print()
print("=" * 60)
print("Example 2: Descriptor grids (electron density + ESP)")
print("=" * 60)

# NOTE: Using RestrictedHF/STO-3G for demonstration purposes only.
# This is a very fast but low-accuracy method. For production results,
# use a higher-level method (e.g., RestrictedHF/cc-pVDZ or DFT).

res = exess.energy(
    TOPOLOGY_FILE,
    method="RestrictedHF",
    basis="STO-3G",
    frag_keywords=None,  # No fragmentation; whole system calc
    export_keywords=exess.ExportKeywords(
        export_density_descriptors=True,
        export_esp_descriptors=True,
        descriptor_grid=exess.RegularDescriptorGrid(
            min=[0.0, 0.0, 0.0],
            max=[1.0, 1.0, 1.0],
            spacing=[1.0, 1.0, 1.0],
        ),
    ),
    convert_hdf5_to_json=True,
    run_spec=RunSpec(storage=1000, gpus=1),
    run_opts=RunOpts(
        name="Rush-Py Tutorial: EXESS Exports 2",
        tags=["rush-py", "tutorial", "exess", "electron density", "ESP"],
    ),
    collect=True,
)

files = exess.save_energy_outputs(res)
print(f"Saved files: {files}")
print()
print("The JSON file contains density_descriptors, esp_descriptors,")
print("descriptor_grid coordinates, and descriptor_grid_weights.")
