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
from pathlib import Path

import numpy as np
from rush import exess
from rush.client import RunOpts, RunSpec

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

# Extract descriptor grid data from the JSON output
grid_data = None
for output in res:
    if 'path' in output and output.get("format") == "Json":
        # First element: flat dict format
        grid_data = output.get("data", {})
        break
    elif 'Json' in output:
        # Second element: type-discriminated format
        grid_data = output['Json'].get("data", {})
        break

if grid_data is None:
    print("WARNING: No JSON output found. Looking for saved JSON files...")
    for f in files:
        if str(f).endswith(".json"):
            with open(f) as jf:
                grid_data = json.load(jf)
            break

# DEBUG: Print structure of extracted grid_data
print(f"\nDEBUG: grid_data type: {type(grid_data)}")
print(f"DEBUG: grid_data keys: {list(grid_data.keys()) if isinstance(grid_data, dict) else 'NOT A DICT'}")
if isinstance(grid_data, dict):
    for key in grid_data.keys():
        val = grid_data[key]
        if isinstance(val, list):
            print(f"  {key}: list of length {len(val)}")
        elif isinstance(val, dict):
            print(f"  {key}: dict with keys {list(val.keys())}")
        else:
            print(f"  {key}: {type(val).__name__}")

if grid_data is None:
    print("ERROR: Could not find grid data. Skipping visualization.")
else:
    # Extract density and ESP values
    density_values = grid_data.get("density_descriptors", [])
    esp_values = grid_data.get("esp_descriptors", [])
    grid_coords = grid_data.get("descriptor_grid", [])

    print(f"  Grid points: {len(density_values)}")
    if density_values:
        print(f"  Density range: [{min(density_values):.6e}, {max(density_values):.6e}]")
    if esp_values:
        print(f"  ESP range: [{min(esp_values):.6e}, {max(esp_values):.6e}]")

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

    # Grid dimensions
    nx = int(round((GRID_MAX[0] - GRID_MIN[0]) / GRID_SPACING[0])) + 1
    ny = int(round((GRID_MAX[1] - GRID_MIN[1]) / GRID_SPACING[1])) + 1
    nz = int(round((GRID_MAX[2] - GRID_MIN[2]) / GRID_SPACING[2])) + 1
    expected_points = nx * ny * nz

    print(f"  Grid dimensions: {nx} × {ny} × {nz} = {expected_points} points")

    if len(density_values) != expected_points:
        print(f"  WARNING: Expected {expected_points} grid points but got {len(density_values)}")
        print("  Attempting to proceed anyway...")

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
    density_arr = np.array(density_values[:expected_points])
    if len(density_arr) < expected_points:
        density_arr = np.pad(density_arr, (0, expected_points - len(density_arr)))
    density_3d = density_arr.reshape((nx, ny, nz))

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

    cube_js = json.dumps(cube_str)
    esp_cube_js = json.dumps(esp_cube_str) if esp_cube_str else "null"
    xyz_js = json.dumps(xyz_str)

    html_content = f"""<!DOCTYPE html>
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

// Add density volume
viewer.addVolumetricData(cubeData, 'cube');
if (espCubeData) {{
  viewer.addVolumetricData(espCubeData, 'cube');
}}

updateSurfaces();
viewer.zoomTo();
viewer.render();

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
  const useESP = document.getElementById('chk-esp').checked && espCubeData;

  if (showDensity) {{
    const surfSpec = {{
      isoval: isoVal,
      smoothness: 3,
      opacity: opacity,
      voldata: cubeData,
      volscheme: useESP ? new $3Dmol.Gradient.RWB(-0.05, 0.05) : undefined,
      volformat: 'cube',
      color: useESP ? undefined : '#3388ff',
    }};

    // If ESP coloring, use the ESP cube as the color source
    if (useESP) {{
      surfSpec.voldata = cubeData;
      surfSpec.volformat = 'cube';
      // Map ESP values to red-white-blue gradient
      viewer.addIsosurface(cubeData, {{
        isoval: isoVal,
        smoothness: 3,
        opacity: opacity,
        volformat: 'cube',
        voldata: espCubeData,
        volscheme: new $3Dmol.Gradient.RWB(-0.05, 0.05),
      }});
    }} else {{
      viewer.addIsosurface(cubeData, {{
        isoval: isoVal,
        smoothness: 3,
        opacity: opacity,
        color: '#4488ff',
        volformat: 'cube',
      }});
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
