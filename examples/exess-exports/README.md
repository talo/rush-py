# EXESS Exports Example

Demonstrates how to export data (electron density, electrostatic potential descriptors) from EXESS energy calculations using rush-py.

## Quick Start

```bash
export RUSH_TOKEN="your-token"
export RUSH_PROJECT="your-project"

# Provide an input topology file, then run:
python 01_exess_exports.py
```

## What This Example Covers

1. Running an EXESS energy calculation with `ExportKeywords`
2. Saving outputs (JSON + HDF5) using `save_energy_outputs`
3. Using `RegularDescriptorGrid` for electron density and ESP descriptors
4. Converting HDF5 exports to JSON with `convert_hdf5_to_json=True`

## Tutorial

See the full tutorial: [EXESS Exports](../../docs/tutorials/exess-exports.md)
