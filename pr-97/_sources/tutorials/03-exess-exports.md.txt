# Tutorial 3: EXESS Exporting Properties

**What you get:** Electron density values, electrostatic potential maps, and other quantum-mechanical properties — exported as HDF5 or JSON files you can analyze with any tool.

| | |
|---|---|
| **Time** | ~1–5 minutes (cloud compute) |
| **Skill level** | Intermediate |
| **Prerequisites** | Python 3.10+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Why This Matters

A single-point energy gives you one number. But the wavefunction EXESS computes contains much more: the **electron density** (where electrons are), the **electrostatic potential** (what a test charge would feel at any point), and other derived quantities. These are the data behind molecular visualization, reactivity prediction, and property-based drug design.

EXESS lets you export these properties on a grid of your choosing — fine-grained for visualization, sparse for quick property extraction, or custom points for specific analyses.

---

## Example 1: Exporting Electron Density

The simplest export: compute an energy and also write out the electron density export file. By default, rush-py converts EXESS exports to JSON. For large export-heavy runs, pass `convert_hdf5_to_json=False` to keep them in their original format.

```python
from rush import exess
from rush import RunOpts

paths = exess.energy(
    "input_topology.json",
    export_keywords=exess.ExportKeywords(export_density=True),
    convert_hdf5_to_json=False,
    run_opts=RunOpts(
        name="Tutorial: EXESS Exports — Density",
        tags=["rush-py", "tutorial", "exess"],
    ),
).save()
print(paths.calc)
print(paths.exports)
```

> ⚠️ **Tutorial Basis Set Warning**
>
> This tutorial [example code](https://github.com/talo/rush-py/tree/main/examples/exess-exports){target="_blank"} uses **STO-3G**, a minimal basis set chosen purely for speed so examples run quickly. **STO-3G is not suitable for research or production calculations.** For meaningful results, use at least `cc-pVDZ` or larger. See the electronic structure methods reference for available basis sets.

### What comes back

EXESS returns a `Run` handle, and `run.save()` turns the outputs into an `exess.ResultPaths` object with named filesystem paths:

```python
# paths looks like:
exess.ResultPaths(
    calc=Path("...json"),
    exports=Path("...hdf5"),
)
```

- **`paths.calc`** — The standard JSON energy output from the EXESS run
- **`paths.exports`** — The exported data file, JSON by default

The HDF5 file can be explored with **h5py**, **h5ls**, **h5dump**, **h5glance**, or any HDF5-compatible tool. Exported data can be large, especially density data on fine grids, so when that's the case it makes sense to obtain the exports in the HDF5 format and save it to disk before loading and reading the data in it.

---

## Example 2: Descriptor Grids — Density and ESP at Specific Points

Often you want property values at *specific locations* — for example, to map the electrostatic potential on a grid around a binding site. EXESS's **descriptor grid** feature lets you define exactly where to sample.

```python
from rush import exess
from rush import RunOpts, RunSpec

paths = exess.energy(
    "input_topology.json",
    frag_keywords=None,  # No fragmentation — whole-system calculation
    export_keywords=exess.ExportKeywords(
        export_density_descriptors=True,
        export_esp_descriptors=True,
        descriptor_grid=exess.RegularDescriptorGrid(
            min=[0.0, 0.0, 0.0],
            max=[1.0, 1.0, 1.0],
            spacing=[1.0, 1.0, 1.0],
        ),
    ),
    run_spec=RunSpec(storage=1000, gpus=1),
    run_opts=RunOpts(
        name="Tutorial: EXESS Exports — Descriptor Grid",
        tags=["rush-py", "tutorial", "exess", "density", "ESP"],
    ),
).save()
```

This computes density and ESP at the 8 corners of a 1 Å cube. For a real molecule like benzene, you'd use a larger grid that envelopes the molecule (e.g., `min=[-5.5, -5.5, -3.5]`, `max=[5.5, 5.5, 3.5]`, `spacing=[0.3, 0.3, 0.3]`). The resulting JSON looks like:

```json
{
  "density_descriptors": [0.0193, 0.0069, 0.0006, 0.2634, 0.0314, 0.0013, 0.0535, 0.0106],
  "esp_descriptors": [-14.74, -12.25, -8.83, -15.69, -12.00, -8.48, -11.79, -9.82],
  "descriptor_grid": [
    [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]
  ],
  "descriptor_grid_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```

**How to read this:**
- `density_descriptors[i]` is the electron density at `descriptor_grid[i]` (in electrons/Å³)
- `esp_descriptors[i]` is the electrostatic potential at `descriptor_grid[i]` (in Hartrees/e)
- High density = lots of electron probability at that point
- Negative ESP = a positive test charge would be attracted there (nucleophilic region)

:::{tip}
For small descriptor grids, obtaining the exports data as JSON is more convenient. For large grids (thousands of points), pass `convert_hdf5_to_json=False` to use HDF5 — the files can be hundreds of MB.
:::

:::{admonition} Fragmentation and descriptor grids
:class: warning
When fragmentation is enabled, descriptor values are stored **per n-mer** (monomer, dimer, trimer), which complicates interpretation. For straightforward grid values, disable fragmentation with `frag_keywords=None` as shown above.
:::

---

## Descriptor Grid Types

EXESS supports several grid definitions:

| Grid type | Use case | Example |
|---|---|---|
| `RegularDescriptorGrid` | Rectangular grid with uniform spacing | `RegularDescriptorGrid(min=[...], max=[...], spacing=[...])` |
| `CustomDescriptorGrid` | Specific points you care about | `CustomDescriptorGrid([x1,y1,z1, x2,y2,z2, ...])` — flat list, keep it small (< hundreds of points) |
| `StandardDescriptorGrid` | Atom-centered radial grids | `StandardDescriptorGrid("ULTRAFINE")` — good default for radial sampling |
| `DescriptorGrid` | Low-level control | `DescriptorGrid(points_per_shell=..., order="One"/"Two", scale=...)` |

**Standard grid options:** `"FINE"`, `"ULTRAFINE"`, `"SUPERFINE"`, `"TREUTLER_GM3"`, `"TREUTLER_GM5"` — these are DFT-style radial quadrature grids centered on atoms.

:::{admonition} Avoid expanded\_esp\_descriptors
:class: danger
The `expanded_esp_descriptors` export currently causes an upstream out-of-memory error bug. Do not enable it.
:::

---

## Example 3: Working with the Output

Once you have the saved files, you can load and inspect them in Python:

:::{tip}
To work with the output directly without needing to save it first, use `run.fetch()` instead of `run.save()`. Here we show how to load the exports data from the saved files. For EXESS exports that can be very large, saving first is often the better workflow.
:::

```python
import json
import h5py

# Load the HDF5 export
with h5py.File(paths.exports, "r") as f:
    print("HDF5 datasets:", list(f.keys()))
    if "density_descriptors" in f:
        density = f["density_descriptors"][:]
        print(f"Density descriptor shape: {density.shape}")
```

Because Example 2 uses the default export behavior, the second file is JSON:

```python
with open(paths.exports) as f:
    export_data = json.load(f)

density = export_data["density_descriptors"]
esp = export_data["esp_descriptors"]
grid = export_data["descriptor_grid"]
print(f"Got {len(density)} density values at {len(grid)} grid points")
```

---

## Visualization

The example script includes additional code to:

- Download and decompress HDF5 output with zstandard
- Interpolate scattered data onto a regular 3D grid with scipy
- Generate an interactive 3D electron density viewer using Three.js Marching Cubes

See the full [`examples/exess-exports/03_exess_exports.py`](https://github.com/talo/rush-py/tree/main/examples/exess-exports){target="_blank"} script for complete details.

### Example Output

Click and drag to rotate the density isosurface. Or run the code above to generate this yourself!

<iframe src="../_static/outputs/density_visualization.html" width="100%" height="600px" frameborder="0"></iframe>

---

## Try It Yourself

The complete example is available in the rush-py repository:

👉 **[examples/exess-exports/](https://github.com/talo/rush-py/tree/main/examples/exess-exports){target="_blank"}**

You'll also need to install extra dependencies:

```bash
pip install h5py zstandard scipy
```

Then download the files or clone the repository to run the example.

The data files (input topology) are included in the repository — no separate download needed.

---

## See Also

- {doc}`Interaction energy tutorial<05-exess-interaction-energy>` — compute energies alongside exports
- {doc}`Optimization tutorial<04-exess-optimization>` — optimize geometry, then export properties
- [Full example script](https://github.com/talo/rush-py/tree/main/examples/exess-exports){target="_blank"}
