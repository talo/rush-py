"""
Example: EXESS Data Exports

This script demonstrates how to:
1. Run an EXESS energy calculation with export keywords
2. Save and inspect the output files
3. Use descriptor grids for electron density and ESP values
4. Generate an interactive 3D visualization of electron density

Tutorial: docs/tutorials/exess-exports.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: benzene_t.json (provided in data/)

Output files (saved to exports-outputs/):
    - density_visualization.html: Interactive 3D electron density viewer
"""

import json
import math
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import zstandard as zstd
from rush import exess
from rush.client import RunOpts, RunSpec, download_object

DATA_DIR = Path(__file__).parent / "data"
TOPOLOGY_FILE = DATA_DIR / "input_topology.json"
OUTPUT_DIR = Path(__file__).parent / "exports-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load topology for later use
with open(TOPOLOGY_FILE) as f:
    topology = json.load(f)

METHOD = "RestrictedHF"
BASIS = "STO-3G"


# ===== Example 1: Basic export with electron density =====
print("=" * 60)
print("Example 1: Exporting electron density")
print("=" * 60)

# NOTE: Using RestrictedHF/STO-3G for demonstration purposes only.
# This is a very fast but low-accuracy method. For production results,
# use a higher-level method (e.g., RestrictedHF/cc-pVDZ or DFT).

res = exess.energy(
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

# Extract total energy from Example 1 results
total_energy = None
for output in res:
    if 'path' in output and output.get("format") == "Json":
        # First element: flat dict format
        json_data = output.get("data", {})
        if "total_energy" in json_data:
            total_energy = json_data["total_energy"]
            break
    elif 'Json' in output:
        # Second element: type-discriminated format
        json_data = output['Json'].get("data", {})
        if "total_energy" in json_data:
            total_energy = json_data["total_energy"]
            break


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

res = exess.energy(
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
    collect=True,
)

files = exess.save_energy_outputs(res)
print(f"Saved files: {files}")
print()
print("The JSON file contains density_descriptors, esp_descriptors,")
print("descriptor_grid coordinates, and descriptor_grid_weights.")

# Try to get total_energy from Example 2 if not found earlier
if total_energy is None:
    for output in res:
        if 'path' in output and output.get("format") == "Json":
            # First element: flat dict format
            json_data = output.get("data", {})
            if "total_energy" in json_data:
                total_energy = json_data["total_energy"]
                break
        elif 'Json' in output:
            # Second element: type-discriminated format
            json_data = output['Json'].get("data", {})
            if "total_energy" in json_data:
                total_energy = json_data["total_energy"]
                break


# ===== Example 3: Generate 3D electron density visualization =====
print()
print("=" * 60)
print("Example 3: 3D Electron Density Visualization")
print("=" * 60)

# Extract descriptor grid data from the HDF5 output
# When convert_hdf5_to_json is NOT set, res[1] contains an HDF5 reference
hdf5_path = None
for output in res:
    if isinstance(output, dict) and 'Hdf5' in output:
        hdf5_path = output['Hdf5']['path']
        break

grid_data = None
if hdf5_path:
    print(f"  Downloading HDF5 from: {hdf5_path}")
    raw_bytes = download_object(hdf5_path)
    
    # Decompress zstandard compression
    print("  Decompressing zstandard archive...")
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(raw_bytes, max_output_size=int(1e9))
    
    # Debug: show what we actually got
    print(f"  Decompressed size: {len(decompressed)} bytes")
    first_bytes = decompressed[:32]
    print(f"  First 32 bytes (hex): {first_bytes.hex()}")
    print(f"  First 32 bytes (repr): {repr(first_bytes)}")
    
    # Check for HDF5 magic
    HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'
    TAR_MAGIC = b'ustar'
    
    is_hdf5 = decompressed.startswith(HDF5_MAGIC)
    is_tar = len(decompressed) > 257 and decompressed[257:262] == TAR_MAGIC
    
    print(f"  Format detection: HDF5={is_hdf5}, TAR={is_tar}")
    
    grid_data = {}
    
    if is_hdf5:
        # Raw HDF5 file (no tar wrapper)
        print("  Reading raw HDF5 file...")
        try:
            with h5py.File(BytesIO(decompressed), "r") as h5f:
                def print_h5_tree(group, indent=0):
                    for key in group.keys():
                        item = group[key]
                        if isinstance(item, h5py.Dataset):
                            print(f"    {'  ' * indent}Dataset '{key}': shape={item.shape}, dtype={item.dtype}")
                        elif isinstance(item, h5py.Group):
                            print(f"    {'  ' * indent}Group '{key}':")
                            print_h5_tree(item, indent + 1)
                
                print("  HDF5 structure:")
                print_h5_tree(h5f)
                
                # Try to extract known keys
                if "density_descriptors" in h5f:
                    grid_data["density_descriptors"] = h5f["density_descriptors"][:].tolist()
                if "esp_descriptors" in h5f:
                    grid_data["esp_descriptors"] = h5f["esp_descriptors"][:].tolist()
                if "descriptor_grid" in h5f:
                    grid_data["descriptor_grid"] = h5f["descriptor_grid"][:].tolist()
                
                # Fallback: if no named keys, use largest float dataset
                if not grid_data:
                    print("  No named datasets found, looking for largest float dataset...")
                    largest = None
                    largest_size = 0
                    for key in h5f.keys():
                        ds = h5f[key]
                        if isinstance(ds, h5py.Dataset) and np.issubdtype(ds.dtype, np.floating):
                            size = ds.size
                            if size > largest_size:
                                largest = key
                                largest_size = size
                    if largest:
                        print(f"  Using dataset '{largest}' (size={largest_size})")
                        grid_data[largest] = h5f[largest][:].tolist()
        except Exception as e:
            print(f"  ERROR reading raw HDF5: {e}")
    
    elif is_tar:
        # Tar archive containing HDF5
        print("  Extracting from tar archive...")
        try:
            with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
                print(f"  Tar members: {[m.name for m in tar.getmembers()]}")
                # Find .h5 files (with or without .h5 extension)
                h5_members = [m for m in tar.getmembers() if m.name.endswith('.h5') or m.isfile()]
                if h5_members:
                    h5_member = h5_members[0]
                    print(f"  Extracting: {h5_member.name}")
                    h5f_obj = tar.extractfile(h5_member)
                    with h5py.File(h5f_obj, "r") as h5f:
                        print(f"  HDF5 datasets: {list(h5f.keys())}")
                        if "density_descriptors" in h5f:
                            grid_data["density_descriptors"] = h5f["density_descriptors"][:].tolist()
                        if "esp_descriptors" in h5f:
                            grid_data["esp_descriptors"] = h5f["esp_descriptors"][:].tolist()
                        if "descriptor_grid" in h5f:
                            grid_data["descriptor_grid"] = h5f["descriptor_grid"][:].tolist()
                else:
                    print("  WARNING: No .h5 files found in tar archive")
        except Exception as e:
            print(f"  ERROR reading tar archive: {e}")
    
    else:
        # Unknown format - search for HDF5 magic
        print("  Unknown format, searching for HDF5 magic in first 10KB...")
        search_bytes = decompressed[:10240]
        hdf5_offset = search_bytes.find(HDF5_MAGIC)
        if hdf5_offset >= 0:
            print(f"  Found HDF5 magic at offset {hdf5_offset}")
            try:
                with h5py.File(BytesIO(decompressed[hdf5_offset:]), "r") as h5f:
                    print(f"  HDF5 datasets: {list(h5f.keys())}")
                    if "density_descriptors" in h5f:
                        grid_data["density_descriptors"] = h5f["density_descriptors"][:].tolist()
                    if "esp_descriptors" in h5f:
                        grid_data["esp_descriptors"] = h5f["esp_descriptors"][:].tolist()
                    if "descriptor_grid" in h5f:
                        grid_data["descriptor_grid"] = h5f["descriptor_grid"][:].tolist()
            except Exception as e:
                print(f"  ERROR reading HDF5 at offset {hdf5_offset}: {e}")
        else:
            print("  ERROR: Could not identify file format (no HDF5 magic found)")
    
    print(f"  Extracted keys from HDF5: {list(grid_data.keys())}")
    for key, val in grid_data.items():
        if isinstance(val, list) and len(val) > 0:
            # Check if values are scalars (1D) or coordinate tuples (2D)
            if isinstance(val[0], (int, float)):
                print(f"    {key}: {len(val)} values, range=[{min(val):.6e}, {max(val):.6e}]")
            else:
                # Coordinate data (e.g., descriptor_grid is list of [x,y,z])
                arr = np.array(val)
                print(f"    {key}: {len(val)} points, shape={arr.shape}")
else:
    # Fallback: look for saved HDF5 files on disk
    print("WARNING: No HDF5 output found in response. Looking for saved files...")
    for f in files:
        if str(f).endswith(".hdf5"):
            with h5py.File(f, "r") as h5f:
                grid_data = {}
                if "density_descriptors" in h5f:
                    grid_data["density_descriptors"] = h5f["density_descriptors"][:].tolist()
                if "esp_descriptors" in h5f:
                    grid_data["esp_descriptors"] = h5f["esp_descriptors"][:].tolist()
                if "descriptor_grid" in h5f:
                    grid_data["descriptor_grid"] = h5f["descriptor_grid"][:].tolist()
            break

if grid_data is None or not grid_data:
    print("ERROR: Could not find grid data. Skipping visualization.")
else:
    # Extract density and ESP values
    density_values = grid_data.get("density_descriptors", [])
    esp_values = grid_data.get("esp_descriptors", [])
    grid_coords = grid_data.get("descriptor_grid", [])

    print(f"  Grid points: {len(density_values)}")
    if density_values:
        dens_arr = np.array(density_values)
        print(f"  Density range (raw): [{dens_arr.min():.6e}, {dens_arr.max():.6e}]")
    
    # Interpolate irregular grid onto regular 3D grid using scipy.interpolate.griddata
    from scipy.interpolate import griddata
    
    if len(density_values) > 0 and len(grid_coords) > 0:
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
        x_min -= padding; x_max += padding
        y_min -= padding; y_max += padding
        z_min -= padding; z_max += padding
        
        # Choose grid resolution (~0.5 bohr spacing, cap at 80 points per axis)
        target_spacing = 0.5  # bohr
        nx = min(80, max(10, int((x_max - x_min) / target_spacing)))
        ny = min(80, max(10, int((y_max - y_min) / target_spacing)))
        nz = min(80, max(10, int((z_max - z_min) / target_spacing)))
        
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        dz = (z_max - z_min) / nz
        
        print(f"  Grid: {nx}×{ny}×{nz} = {nx*ny*nz} points")
        print(f"  Bounds: x=[{x_min:.2f},{x_max:.2f}] y=[{y_min:.2f},{y_max:.2f}] z=[{z_min:.2f},{z_max:.2f}]")
        print(f"  Spacing: ({dx:.4f}, {dy:.4f}, {dz:.4f}) bohr")
        
        # Create regular grid
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        zi = np.linspace(z_min, z_max, nz)
        
        grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        
        # Interpolate using linear method, fill outside convex hull with NaN
        # so we can diagnose how many points are outside the convex hull
        print(f"  Interpolating (linear, fill_value=NaN for diagnosis)...")
        grid_values = griddata(coords, values, grid_points, method='linear', fill_value=np.nan)
        
        # ============ DEBUG DIAGNOSTICS ============
        total = len(grid_values)
        nan_count = np.sum(np.isnan(grid_values))
        finite_count = total - nan_count
        print(f"\n  🔍 DEBUG: Interpolation diagnostics:")
        print(f"  Total grid points: {total}")
        print(f"  NaN (outside convex hull): {nan_count} ({100*nan_count/total:.1f}%)")
        print(f"  Finite (inside convex hull): {finite_count} ({100*finite_count/total:.1f}%)")
        
        if finite_count > 0:
            finite_vals = grid_values[np.isfinite(grid_values)]
            print(f"  Finite value range: [{finite_vals.min():.6e}, {finite_vals.max():.6e}]")
            print(f"  Finite value mean: {finite_vals.mean():.6e}")
            print(f"  Finite value median: {np.median(finite_vals):.6e}")
            
            # Distribution analysis
            neg_count = np.sum(finite_vals < 0)
            tiny_count = np.sum((finite_vals >= 0) & (finite_vals < 1e-10))
            small_count = np.sum((finite_vals >= 1e-10) & (finite_vals < 1e-5))
            medium_count = np.sum((finite_vals >= 1e-5) & (finite_vals < 1e-1))
            large_count = np.sum(finite_vals >= 1e-1)
            
            print(f"  Distribution of finite values:")
            print(f"    Negative:        {neg_count}")
            print(f"    [0, 1e-10):      {tiny_count}")
            print(f"    [1e-10, 1e-5):   {small_count}")
            print(f"    [1e-5, 1e-1):    {medium_count}")
            print(f"    [1e-1, ∞):       {large_count}")
            
            # Percentiles of finite values
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                print(f"    P{p:2d}: {np.percentile(finite_vals, p):.6e}")
        
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
                print(f"  Noise threshold (adaptive): {noise_threshold:.6e}")
                noise_count = np.sum((grid_values > 0) & (grid_values < noise_threshold))
                grid_values[(grid_values > 0) & (grid_values < noise_threshold)] = 0.0
                print(f"  Noise points clipped: {noise_count}")
        
        # Also zero out negative interpolation artifacts
        neg_artifacts = np.sum(grid_values < 0)
        if neg_artifacts > 0:
            print(f"  Negative artifacts zeroed: {neg_artifacts}")
            grid_values[grid_values < 0] = 0.0
        
        # Final statistics
        zero_count = np.sum(grid_values == 0.0)
        nonzero_count = total - zero_count
        
        grid_values = grid_values.astype(np.float32)
        print(f"\n  ✓ Final grid range: [{grid_values.min():.3e}, {grid_values.max():.3e}]")
        print(f"  Points with data: {nonzero_count} ({100*nonzero_count/total:.1f}%)")
        print(f"  Points zero: {zero_count} ({100*zero_count/total:.1f}%)")
        
        # Suggest isovalues
        if nonzero_count > 0:
            nz_vals = grid_values[grid_values > 0]
            print(f"  Suggested isovalues:")
            for pct, label in [(90, "thick surface"), (50, "medium"), (10, "thin/outer")]:
                iso = np.percentile(nz_vals, pct)
                print(f"    {label} (P{pct}): {iso:.3e}")
        
        density_3d = grid_values.reshape((nx, ny, nz))
        
        # Convert bounds from bohr to angstrom for cube file
        ANG_TO_BOHR = 1.8897259886
        GRID_MIN = [x_min / ANG_TO_BOHR, y_min / ANG_TO_BOHR, z_min / ANG_TO_BOHR]
        GRID_SPACING = [dx / ANG_TO_BOHR, dy / ANG_TO_BOHR, dz / ANG_TO_BOHR]
    else:
        print("ERROR: No grid data to interpolate. Skipping visualization.")
        density_3d = None
    
    if density_3d is not None:
        # ---- Build Gaussian Cube file from grid data ----
        # Cube format: https://gaussian.com/cubegen/
        # 3Dmol.js can directly parse cube files for isosurface rendering

        # Angstrom to Bohr conversion
        ANG_TO_BOHR = 1.8897259886

        # Atomic numbers lookup
        ATOMIC_NUMBERS = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
            "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
            "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
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
        cube_lines.append(f"{n_atoms:5d} {origin_bohr[0]:12.6f} {origin_bohr[1]:12.6f} {origin_bohr[2]:12.6f}")
        # Number of voxels along each axis and step vector
        cube_lines.append(f"{nx:5d} {spacing_bohr[0]:12.6f} {0.0:12.6f} {0.0:12.6f}")
        cube_lines.append(f"{ny:5d} {0.0:12.6f} {spacing_bohr[1]:12.6f} {0.0:12.6f}")
        cube_lines.append(f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {spacing_bohr[2]:12.6f}")
        # Atom lines
        for i in range(n_atoms):
            at_num = ATOMIC_NUMBERS.get(symbols[i], 0)
            x_b = geometry[3*i] * ANG_TO_BOHR
            y_b = geometry[3*i+1] * ANG_TO_BOHR
            z_b = geometry[3*i+2] * ANG_TO_BOHR
            cube_lines.append(f"{at_num:5d} {float(at_num):12.6f} {x_b:12.6f} {y_b:12.6f} {z_b:12.6f}")

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
            esp_lines = cube_lines[:6 + n_atoms]  # reuse header
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
        cube_path.write_text(cube_str)
        print(f"  ✓ Cube file saved: {cube_path}")

        if esp_cube_str:
            esp_cube_path = OUTPUT_DIR / "esp.cube"
            esp_cube_path.write_text(esp_cube_str)
            print(f"  ✓ ESP cube file saved: {esp_cube_path}")

        # ---- Build XYZ string for 3Dmol.js ----
        xyz_lines = [str(n_atoms), f"{METHOD}/{BASIS} benzene"]
        for i in range(n_atoms):
            xyz_lines.append(
                f"{symbols[i]}  {geometry[3*i]:.6f}  {geometry[3*i+1]:.6f}  {geometry[3*i+2]:.6f}"
            )
        xyz_str = "\n".join(xyz_lines)

        # ---- Generate interactive HTML ----
        energy_display = f"{total_energy:.8f} Eh" if total_energy is not None else "N/A"
        energy_kcal = f"{total_energy * 627.509474:.2f} kcal/mol" if total_energy is not None else ""

        # The inline 3Dmol fallback template is kept below but only evaluated
        # when viewer_template.html is missing (see conditional after it).
        # Skip the f-string entirely — jump straight to the template approach.
        html_content_template = None  # placeholder, used only in fallback

        if False:  # --- BEGIN DEAD 3Dmol FALLBACK (kept for reference) ---
            html_content_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Electron Density Visualization — Benzene</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f0f1a; color: #e0e0e0; }}
  .header {{
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 28px 40px;
    border-bottom: 2px solid #0f3460;
  }}
  .header h1 {{ font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; color: #fff; }}
  .header p {{ opacity: 0.6; margin-top: 4px; font-size: 0.85rem; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

  .summary {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 14px; margin-bottom: 24px;
  }}
  .stat-card {{
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 10px;
    padding: 16px 20px;
  }}
  .stat-card .label {{
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #8888aa; margin-bottom: 4px;
  }}
  .stat-card .value {{
    font-size: 1.1rem; font-weight: 600; color: #e0e0ff;
    font-variant-numeric: tabular-nums;
  }}

  .main-panel {{
    display: grid; grid-template-columns: 1fr 300px; gap: 20px;
  }}
  .viewer-panel {{
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px;
    overflow: hidden;
  }}
  .viewer-header {{
    padding: 14px 20px; border-bottom: 1px solid #2a2a4a;
    font-weight: 600; font-size: 0.9rem; color: #aab;
  }}
  #viewer-container {{ width: 100%; height: 550px; }}

  .controls-panel {{
    background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px;
    padding: 20px; display: flex; flex-direction: column; gap: 18px;
  }}
  .controls-panel h3 {{
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: #8888aa; margin-bottom: 8px;
  }}
  .control-group {{ display: flex; flex-direction: column; gap: 8px; }}
  .control-row {{
    display: flex; align-items: center; justify-content: space-between;
  }}
  .control-row label {{ font-size: 0.85rem; color: #ccc; }}
  .control-row input[type="range"] {{ width: 120px; }}
  .control-row .val {{ font-size: 0.75rem; color: #8888aa; min-width: 60px; text-align: right; }}

  .btn {{
    padding: 8px 16px; border: 1px solid #3a3a5a; border-radius: 6px;
    background: #2a2a4a; color: #ddd; cursor: pointer; font-size: 0.8rem;
    transition: all 0.15s;
  }}
  .btn:hover {{ background: #3a3a6a; border-color: #5a5a8a; }}
  .btn.active {{ background: #0f3460; border-color: #1a5a9a; color: #fff; }}

  .btn-group {{ display: flex; gap: 6px; flex-wrap: wrap; }}

  .footer {{
    margin-top: 24px; text-align: center; font-size: 0.75rem;
    color: #555; padding: 16px;
  }}
</style>
</head>
<body>
<div class="header">
  <h1>🔬 Electron Density &amp; ESP Visualization</h1>
  <p>Benzene (C₆H₆) — {METHOD}/{BASIS}</p>
</div>
<div class="container">

  <div class="summary">
    <div class="stat-card">
      <div class="label">Method / Basis</div>
      <div class="value">{METHOD} / {BASIS}</div>
    </div>
    <div class="stat-card">
      <div class="label">Total Energy</div>
      <div class="value">{energy_display}</div>
    </div>
    <div class="stat-card">
      <div class="label">Grid Points</div>
      <div class="value">{expected_points:,}</div>
    </div>
    <div class="stat-card">
      <div class="label">Grid Spacing</div>
      <div class="value">{GRID_SPACING[0]} Å</div>
    </div>
  </div>

  <div class="main-panel">
    <div class="viewer-panel">
      <div class="viewer-header">Interactive 3D Viewer — click &amp; drag to rotate, scroll to zoom</div>
      <div id="viewer-container"></div>
    </div>

    <div class="controls-panel">
      <div class="control-group">
        <h3>Isosurface</h3>
        <div class="control-row">
          <label>Show density</label>
          <input type="checkbox" id="chk-density" checked onchange="updateSurfaces()">
        </div>
        <div class="control-row">
          <label>Isovalue</label>
          <input type="range" id="iso-slider" min="-5" max="-1" step="0.1" value="-3"
                 oninput="updateIsoLabel(); updateSurfaces()">
          <span class="val" id="iso-label">0.001</span>
        </div>
        <div class="control-row">
          <label>Opacity</label>
          <input type="range" id="opacity-slider" min="0.1" max="1.0" step="0.05" value="0.6"
                 oninput="updateSurfaces()">
        </div>
      </div>

      <div class="control-group">
        <h3>ESP Coloring</h3>
        <div class="control-row">
          <label>Color by ESP</label>
          <input type="checkbox" id="chk-esp" onchange="updateSurfaces()"
                 {"" if esp_cube_str else 'disabled title="No ESP data available"'}>
        </div>
        <div style="font-size:0.75rem; color:#888; margin-top:4px;">
          {"🔴 Negative (nucleophilic) → 🔵 Positive (electrophilic)" if esp_cube_str else "ESP data not available for this run"}
        </div>
      </div>

      <div class="control-group">
        <h3>Molecule Style</h3>
        <div class="btn-group">
          <button class="btn active" id="btn-ballstick" onclick="setStyle('ballstick')">Ball &amp; Stick</button>
          <button class="btn" id="btn-stick" onclick="setStyle('stick')">Stick</button>
          <button class="btn" id="btn-sphere" onclick="setStyle('sphere')">Space Fill</button>
          <button class="btn" id="btn-wire" onclick="setStyle('wire')">Wire</button>
        </div>
      </div>

      <div class="control-group">
        <h3>Background</h3>
        <div class="btn-group">
          <button class="btn active" onclick="viewer.setBackgroundColor('#0f0f1a'); viewer.render();">Dark</button>
          <button class="btn" onclick="viewer.setBackgroundColor('#ffffff'); viewer.render();">White</button>
          <button class="btn" onclick="viewer.setBackgroundColor('#000000'); viewer.render();">Black</button>
        </div>
      </div>

      <div class="control-group">
        <h3>View</h3>
        <div class="btn-group">
          <button class="btn" onclick="viewer.zoomTo(); viewer.render();">Reset View</button>
          <button class="btn" onclick="viewer.spin('y'); spinning=!spinning;" id="btn-spin">Spin</button>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    Generated by Rush-Py EXESS Exports example &bull; Powered by 3Dmol.js
  </div>
</div>

<script>
const cubeData = {cube_js};
const espCubeData = {esp_cube_js};
const xyzData = {xyz_js};

let viewer = $3Dmol.createViewer('viewer-container', {{
  backgroundColor: '#0f0f1a',
  antialias: true,
}});
let spinning = false;

// Add molecule
viewer.addModel(xyzData, 'xyz');
setStyle('ballstick');

// Parse and add cube volumes
let vol = null;
try {{
  // Parse cube file manually
  vol = parseCubeToVolume(cubeData);
}} catch(e) {{
  console.error('Error parsing cube:', e);
}}

updateSurfaces();
viewer.zoomTo();
viewer.render();

// Simple cube file parser
function parseCubeToVolume(cubeStr) {{
  const lines = cubeStr.trim().split('\\n');
  // Skip comment lines and parse header
  const header = lines[2].split(/\\s+/).map(Number);
  const natoms = header[0];
  
  // Parse grid parameters
  const gridX = lines[3].split(/\\s+/).slice(0, 4).map(Number);
  const gridY = lines[4].split(/\\s+/).slice(0, 4).map(Number);
  const gridZ = lines[5].split(/\\s+/).slice(0, 4).map(Number);
  
  const nx = gridX[0], ny = gridY[0], nz = gridZ[0];
  const xstep = gridX[1], ystep = gridY[2], zstep = gridZ[3];
  
  // Skip atom lines and parse volumetric data
  const dataStart = 6 + natoms;
  const data = [];
  for (let i = dataStart; i < lines.length; i++) {{
    const vals = lines[i].trim().split(/\\s+/).map(Number);
    data.push(...vals);
  }}
  
  return {{
    origin: [header[1], header[2], header[3]],
    nX: nx, nY: ny, nZ: nz,
    data: new Float32Array(data),
    ystep: ystep,
    zstep: zstep,
    xstep: xstep
  }};
}}

function updateIsoLabel() {{
  const slider = document.getElementById('iso-slider');
  const val = Math.pow(10, parseFloat(slider.value));
  document.getElementById('iso-label').textContent = val.toExponential(1);
}}

function updateSurfaces() {{
  viewer.removeAllSurfaces();
  const showDensity = document.getElementById('chk-density').checked;
  const isoVal = Math.pow(10, parseFloat(document.getElementById('iso-slider').value));
  const opacity = parseFloat(document.getElementById('opacity-slider').value);
  
  if (showDensity && vol) {{
    try {{
      // Add volume surface with the parsed cube data
      viewer.addSurface($3Dmol.SurfaceType.VDW, {{
        voldata: vol,
        isoval: isoVal,
        opacity: opacity,
        color: '#4488ff',
        smoothness: 2
      }});
    }} catch(e) {{
      console.error('Error adding surface:', e);
    }}
  }}
  viewer.render();
}}

function setStyle(style) {{
  document.querySelectorAll('.btn-group .btn').forEach(b => b.classList.remove('active'));
  const btnId = 'btn-' + (style === 'ballstick' ? 'ballstick' : style);
  const btn = document.getElementById(btnId);
  if (btn) btn.classList.add('active');

  switch(style) {{
    case 'ballstick':
      viewer.setStyle({{}}, {{
        stick: {{ radius: 0.14, colorscheme: 'Jmol' }},
        sphere: {{ scale: 0.28, colorscheme: 'Jmol' }}
      }});
      break;
    case 'stick':
      viewer.setStyle({{}}, {{ stick: {{ colorscheme: 'Jmol' }} }});
      break;
    case 'sphere':
      viewer.setStyle({{}}, {{ sphere: {{ colorscheme: 'Jmol' }} }});
      break;
    case 'wire':
      viewer.setStyle({{}}, {{ line: {{ colorscheme: 'Jmol' }} }});
      break;
  }}
  viewer.render();
}}
</script>
</body>
</html>"""
        # --- END DEAD 3Dmol FALLBACK ---

        # ---- Use Three.js Marching Cubes viewer ----
        # Load the template and embed the cube data
        template_path = Path(__file__).parent / "viewer_template.html"
        if template_path.exists():
            with open(template_path) as f:
                html_template = f.read()
            # Escape cube data for embedding in JavaScript template literal
            cube_text_escaped = cube_str.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
            html_content = html_template.replace('__CUBE_DATA__', cube_text_escaped)
            print(f"  ✓ Using Three.js Marching Cubes viewer (fast, beautiful isosurface rendering)")
        else:
            print(f"  ⚠ viewer_template.html not found at {template_path}")
            html_content = "<html><body><h1>Error: viewer_template.html not found</h1></body></html>"

        html_path = OUTPUT_DIR / "density_visualization.html"
        html_path.write_text(html_content)
        print(f"  ✓ Visualization saved: {html_path}")
        print(f"  Open in a browser to explore the electron density isosurface!")
    print()
    print("Features:")
    print("  • Rotate: click & drag | Zoom: scroll | Pan: right-click drag")
    print("  • Adjust isosurface threshold with the slider")
    print("  • Toggle ESP coloring to see electrostatic potential on the surface")
    print("  • Switch molecule rendering styles")
