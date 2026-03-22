"""
Example: EXESS Data Exports

This script demonstrates how to:
1. Run an EXESS energy calculation with export keywords
2. Save and inspect the output files
3. Use descriptor grids for electron density and ESP values
4. Generate an interactive 3D visualization of electron density

Tutorial: https://exess.qdx.co/docs/tutorials/03-exess-exports.html

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: benzene_t.json (provided in data/)

Output files (saved to exports-outputs/):
    - density_visualization.html: Interactive 3D electron density viewer
"""

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata

from rush import exess
from rush.client import RunOpts, RunSpec
from rush.exess import energy

DATA_DIR = Path(__file__).parent / "data"
TOPOLOGY_FILE = DATA_DIR / "input_topology.json"
OUTPUT_DIR = Path(__file__).parent / "exports-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load topology for later use
with open(TOPOLOGY_FILE, encoding="utf-8") as f:
    topology = json.load(f)

METHOD = "RestrictedHF"
BASIS = "STO-3G"


# ===== Example 1: Basic export with electron density =====
print("=" * 60)
print("Example 1: Exporting electron density")
print("=" * 60)

# ⚠️ TUTORIAL ONLY: STO-3G is a minimal basis set used here for speed/demonstration.
# It is NOT suitable for research or production use. For real work, use at least
# cc-pVDZ or larger (e.g., cc-pVTZ, aug-cc-pVDZ) with an appropriate method.

result = energy(
    TOPOLOGY_FILE,
    method=METHOD,
    basis=BASIS,
    export_keywords=exess.ExportKeywords(
        export_density=True,
    ),
    run_opts=RunOpts(
        name="Rush-Py Tutorial: EXESS Exports 1",
        tags=["rush-py", "tutorial", "exess"],
    ),
).collect()

# Inspect the result
print(f"Calc object: {result.calc}")
print(f"Exports object: {result.exports}")

# Save outputs to disk (JSON + HDF5)
paths = result.save()
print(f"Saved files: {paths}")

# Load total energy from fetched outputs
res = result.fetch()
total_energy = res.calc.qmmbe.expanded_hf_energy


# ===== Example 2: Descriptor grids for density and ESP =====
print()
print("=" * 60)
print("Example 2: Descriptor grids (electron density + ESP)")
print("=" * 60)

# Use a meaningful grid that envelopes the benzene molecule.
# Benzene extends roughly ±2.5 Å in x/y, flat in z.
# We pad by ~3 Å and use 0.3 Å spacing for reasonable resolution.
GRID_MIN = [-5.5, -5.5, -3.5]
GRID_MAX = [5.5, 5.5, 3.5]
GRID_SPACING = [0.3, 0.3, 0.3]

result = energy(
    TOPOLOGY_FILE,
    method=METHOD,
    basis=BASIS,
    frag_keywords=None,  # No fragmentation; whole system calc
    export_keywords=exess.ExportKeywords(
        export_density_descriptors=True,
        export_esp_descriptors=True,
        descriptor_grid=exess.RegularDescriptorGrid(
            min=GRID_MIN,
            max=GRID_MAX,
            spacing=GRID_SPACING,
        ),
    ),
    run_spec=RunSpec(storage=1000, gpus=1),
    run_opts=RunOpts(
        name="Rush-Py Tutorial: EXESS Exports 2",
        tags=["rush-py", "tutorial", "exess", "electron density", "ESP"],
    ),
).collect()

paths = result.save()
print(f"Saved files: {paths}")
print()
print("The JSON file contains density_descriptors, esp_descriptors,")
print("descriptor_grid coordinates, and descriptor_grid_weights.")


# ===== Example 3: Generate 3D electron density visualization =====
print()
print("=" * 60)
print("Example 3: 3D Electron Density Visualization")
print("=" * 60)

# Load descriptor grid data from the saved HDF5 file
print("  Loading HDF5 from saved files...")
grid_data = None

with h5py.File(paths.exports, "r") as h5f:
    grid_data = {}
    # Extract known keys
    if "density_descriptors" in h5f:
        grid_data["density_descriptors"] = h5f["density_descriptors"][:].tolist()
    if "esp_descriptors" in h5f:
        grid_data["esp_descriptors"] = h5f["esp_descriptors"][:].tolist()
    if "descriptor_grid" in h5f:
        grid_data["descriptor_grid"] = h5f["descriptor_grid"][:].tolist()

print(f"  ✓ Extracted keys from HDF5: {list(grid_data.keys())}")
for key, val in grid_data.items():
    if isinstance(val, list) and len(val) > 0:
        # Check if values are scalars (1D) or coordinate tuples (2D)
        if isinstance(val[0], (int, float)):
            print(
                f"    {key}: {len(val)} values, range=[{min(val):.6e}, {max(val):.6e}]"
            )
        else:
            # Coordinate data (e.g., descriptor_grid is list of [x,y,z])
            arr = np.array(val)
            print(f"    {key}: {len(val)} points, shape={arr.shape}")

# Extract density and ESP values
density_values = grid_data.get("density_descriptors", [])
esp_values = grid_data.get("esp_descriptors", [])
grid_coords = grid_data.get("descriptor_grid", [])

print(f"  Grid points: {len(density_values)}")
if density_values:
    dens_arr = np.array(density_values)
    print(f"  Density range (raw): [{dens_arr.min():.6e}, {dens_arr.max():.6e}]")

# Interpolate irregular grid onto regular 3D grid using scipy.interpolate.griddata
print(f"\n  Interpolating {len(density_values)} surface points onto regular 3D grid...")

coords = np.array(grid_coords)
values = np.array(density_values)

# Reshape coords if needed (should be Nx3)
if coords.ndim == 1:
    coords = coords.reshape(-1, 3)
elif coords.ndim == 2 and coords.shape[0] == 3 and coords.shape[1] > 3:
    # HDF5 stores as (3, N) format: [x_coords, y_coords, z_coords]
    # Transpose to (N, 3) format: [[x,y,z], [x,y,z], ...]
    coords = coords.T

# Determine grid bounds from the point cloud
x_min, y_min, z_min = coords.min(axis=0)
x_max, y_max, z_max = coords.max(axis=0)

# Add fixed padding (3 bohr ≈ 1.6 Å beyond molecular extent)
padding = 3.0  # bohr
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding
z_min -= padding
z_max += padding

# Choose grid resolution (~0.5 bohr spacing, cap at 80 points per axis)
target_spacing = 0.5  # bohr
nx = min(80, max(10, int((x_max - x_min) / target_spacing)))
ny = min(80, max(10, int((y_max - y_min) / target_spacing)))
nz = min(80, max(10, int((z_max - z_min) / target_spacing)))

dx = (x_max - x_min) / nx
dy = (y_max - y_min) / ny
dz = (z_max - z_min) / nz

print(f"  Grid: {nx}×{ny}×{nz} = {nx * ny * nz} points")
print(
    f"  Bounds: x=[{x_min:.2f},{x_max:.2f}] y=[{y_min:.2f},{y_max:.2f}] z=[{z_min:.2f},{z_max:.2f}]"
)
print(f"  Spacing: ({dx:.4f}, {dy:.4f}, {dz:.4f}) bohr")

# Create regular grid
xi = np.linspace(x_min, x_max, nx)
yi = np.linspace(y_min, y_max, ny)
zi = np.linspace(z_min, z_max, nz)

grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing="ij")
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

# Interpolate using linear method, fill outside convex hull with NaN
# so we can diagnose how many points are outside the convex hull
print("  Interpolating (linear, fill_value=NaN for diagnosis)...")
grid_values = griddata(coords, values, grid_points, method="linear", fill_value=np.nan)

# ============ Grid statistics ============
total = len(grid_values)
nan_count = np.sum(np.isnan(grid_values))
finite_count = total - nan_count

# Now set NaN → 0.0 (density outside molecular surface is zero)
grid_values = np.where(np.isnan(grid_values), 0.0, grid_values)

# Clip noise: use percentile-based threshold instead of fixed 1e-10
# The old threshold of 1e-10 was wiping ALL data if values are very small!
if finite_count > 0:
    finite_vals = grid_values[grid_values != 0.0]
    if len(finite_vals) > 0:
        # Use 1st percentile of positive values as noise floor
        pos_vals = finite_vals[finite_vals > 0]
        if len(pos_vals) > 0:
            noise_floor = np.percentile(pos_vals, 1)
            # But never clip above 1e-5 (safety)
            noise_threshold = min(noise_floor * 0.1, 1e-5)
        else:
            noise_threshold = 0.0  # no positive values, skip clipping
        grid_values[(grid_values > 0) & (grid_values < noise_threshold)] = 0.0

# Also zero out negative interpolation artifacts
grid_values[grid_values < 0] = 0.0

# Final statistics
zero_count = np.sum(grid_values == 0.0)
nonzero_count = total - zero_count

grid_values = grid_values.astype(np.float32)
print(f"\n  ✓ Final grid range: [{grid_values.min():.3e}, {grid_values.max():.3e}]")
print(f"  Points with data: {nonzero_count} ({100 * nonzero_count / total:.1f}%)")
print(f"  Points zero: {zero_count} ({100 * zero_count / total:.1f}%)")

# Calculate percentiles for visualization defaults
iso_p1 = None
iso_p10 = None
iso_p90 = None
if nonzero_count > 0:
    nz_vals = grid_values[grid_values > 0]
    iso_p1 = np.percentile(nz_vals, 1)
    iso_p10 = np.percentile(nz_vals, 10)
    iso_p90 = np.percentile(nz_vals, 90)

density_3d = grid_values.reshape((nx, ny, nz))

# Convert bounds from bohr to angstrom for cube file
ANG_TO_BOHR = 1.8897259886
GRID_MIN = [x_min / ANG_TO_BOHR, y_min / ANG_TO_BOHR, z_min / ANG_TO_BOHR]
GRID_SPACING = [dx / ANG_TO_BOHR, dy / ANG_TO_BOHR, dz / ANG_TO_BOHR]

# ---- Build Gaussian Cube file from grid data ----
# Cube format: https://gaussian.com/cubegen/
# 3Dmol.js can directly parse cube files for isosurface rendering

# Angstrom to Bohr conversion
ANG_TO_BOHR = 1.8897259886

# Atomic numbers lookup
ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}

symbols = topology["symbols"]
geometry = topology["geometry"]  # flat list: [x0,y0,z0, x1,y1,z1, ...]
n_atoms = len(symbols)

expected_points = nx * ny * nz
print(f"  Building cube file: {nx} × {ny} × {nz} = {expected_points} points")

# Build the cube file string
origin_bohr = [v * ANG_TO_BOHR for v in GRID_MIN]
spacing_bohr = [v * ANG_TO_BOHR for v in GRID_SPACING]

cube_lines = []
cube_lines.append("Electron Density")
cube_lines.append(f"Generated by Rush-Py EXESS Exports ({METHOD}/{BASIS})")
# Number of atoms, origin
cube_lines.append(
    f"{n_atoms:5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}"
)
# Number of voxels along each axis and step vector
cube_lines.append(f"{nx:5d} {spacing_bohr[0]:12.6f} {0.0:12.6f} {0.0:12.6f}")
cube_lines.append(f"{ny:5d} {0.0:12.6f} {spacing_bohr[1]:12.6f} {0.0:12.6f}")
cube_lines.append(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {spacing_bohr[2]:12.6f}")
# Atom lines
for i in range(n_atoms):
    at_num = ATOMIC_NUMBERS.get(symbols[i], 0)
    x_b = geometry[3 * i] * ANG_TO_BOHR
    y_b = geometry[3 * i + 1] * ANG_TO_BOHR
    z_b = geometry[3 * i + 2] * ANG_TO_BOHR
    cube_lines.append(
        f"{at_num:5d} {float(at_num):12.6f} {x_b:12.6f} {y_b:12.6f} {z_b:12.6f}"
    )

# Volumetric data (fast axis = z, then y, then x — Cube convention)
# Reshape density to 3D array and write in Cube order
for ix in range(nx):
    for iy in range(ny):
        row_vals = []
        for iz in range(nz):
            row_vals.append(f"{density_3d[ix, iy, iz]:13.5e}")
            if len(row_vals) == 6:
                cube_lines.append(" ".join(row_vals))
                row_vals = []
        if row_vals:
            cube_lines.append(" ".join(row_vals))

cube_str = "\n".join(cube_lines)

# Also build ESP cube if available
esp_cube_str = None
if esp_values and len(esp_values) >= expected_points:
    esp_lines = cube_lines[: 6 + n_atoms]  # reuse header
    esp_lines[0] = "Electrostatic Potential"
    esp_arr = np.array(esp_values[:expected_points]).reshape((nx, ny, nz))
    for ix in range(nx):
        for iy in range(ny):
            row_vals = []
            for iz in range(nz):
                row_vals.append(f"{esp_arr[ix, iy, iz]:13.5e}")
                if len(row_vals) == 6:
                    esp_lines.append(" ".join(row_vals))
                    row_vals = []
            if row_vals:
                esp_lines.append(" ".join(row_vals))
    esp_cube_str = "\n".join(esp_lines)

# Save cube files
cube_path = OUTPUT_DIR / "electron_density.cube"
cube_path.write_text(cube_str, encoding="utf-8")
print(f"  ✓ Cube file saved: {cube_path}")

if esp_cube_str:
    esp_cube_path = OUTPUT_DIR / "esp.cube"
    esp_cube_path.write_text(esp_cube_str, encoding="utf-8")
    print(f"  ✓ ESP cube file saved: {esp_cube_path}")

# ---- Build XYZ string for 3Dmol.js ----
xyz_lines = [str(n_atoms), f"{METHOD}/{BASIS} benzene"]
for i in range(n_atoms):
    xyz_lines.append(
        f"{symbols[i]}  {geometry[3 * i]:.6f}  {geometry[3 * i + 1]:.6f}  {geometry[3 * i + 2]:.6f}"
    )
xyz_str = "\n".join(xyz_lines)

# ---- Generate interactive HTML ----
energy_display = f"{total_energy:.8f} Eh" if total_energy is not None else "N/A"
energy_kcal = (
    f"{total_energy * 627.509474:.2f} kcal/mol" if total_energy is not None else ""
)

# ---- Use Three.js Marching Cubes viewer ----
# Load the template and embed the cube data
template_path = Path(__file__).parent / "viewer_template.html"
if template_path.exists():
    with open(template_path, encoding="utf-8") as f:
        html_template = f.read()
    # Escape cube data for embedding in JavaScript template literal
    cube_text_escaped = (
        cube_str.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    )
    html_content = html_template.replace("__CUBE_DATA__", cube_text_escaped)

    # Set isovalue slider defaults to show full outer electron density
    # Use P1 as min (noise floor), P10 as default (full cloud), P90 as max (dense core)
    if iso_p1 is not None and iso_p10 is not None and iso_p90 is not None:
        html_content = html_content.replace("__ISO_MIN__", f"{iso_p1:.6e}")
        html_content = html_content.replace("__ISO_DEFAULT__", f"{iso_p10:.6e}")
        html_content = html_content.replace("__ISO_MAX__", f"{iso_p90:.6e}")

    print(
        "  ✓ Using Three.js Marching Cubes viewer (fast, beautiful isosurface rendering)"
    )
else:
    print(f"  ⚠ viewer_template.html not found at {template_path}")
    html_content = (
        "<html><body><h1>Error: viewer_template.html not found</h1></body></html>"
    )

html_path = OUTPUT_DIR / "density_visualization.html"
html_path.write_text(html_content, encoding="utf-8")
print(f"  ✓ Visualization saved: {html_path}")
print("  Open in a browser to explore the electron density isosurface!")
print()
print("Features:")
print("  • Rotate: click & drag | Zoom: scroll | Pan: right-click drag")
print("  • Adjust isosurface threshold with the slider")
print("  • Toggle ESP coloring to see electrostatic potential on the surface")
print("  • Switch molecule rendering styles")
