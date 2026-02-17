"""
Example: EXESS Data Exports

This script demonstrates how to:
1. Run an EXESS energy calculation with export keywords
2. Save and inspect the output files
3. Use descriptor grids for electron density and ESP values

Tutorial: docs/tutorials/exess-exports.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Provide an input topology file (e.g., from tests/data/)
"""

from rush import exess
from rush.client import RunOpts, RunSpec


# ===== Example 1: Basic export with electron density =====
print("=" * 60)
print("Example 1: Exporting electron density")
print("=" * 60)

res = exess.energy(
    "input_topology.json",
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
    print(f"  [{i}] path={output['path']}, format={output['format']}")

# Save outputs to disk (JSON + HDF5)
files = exess.save_energy_outputs(res)
print(f"Saved files: {files}")


# ===== Example 2: Descriptor grids for density and ESP =====
print()
print("=" * 60)
print("Example 2: Descriptor grids (electron density + ESP)")
print("=" * 60)

res = exess.energy(
    "input_topology.json",
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
