"""
Example: EXESS Single Point Energy Calculation

This script demonstrates how to:
1. Run a single-point energy (SPE) calculation using Rush
2. Extract the total energy and electronic properties
3. Generate a simple HTML visualization of the results

Tutorial: docs/tutorials/02-exess-spe.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: water_topology.json (provided in data/)

Output files (saved to spe-outputs/):
    - spe_results.html: Interactive 3D molecule view + energy summary
"""

import json
from pathlib import Path

from rush import exess
from rush.client import RunOpts

DATA_DIR = Path(__file__).parent / "data"
TOPOLOGY_FILE = DATA_DIR / "water_topology.json"
OUTPUT_DIR = Path(__file__).parent / "spe-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load topology for visualization later
with open(TOPOLOGY_FILE) as f:
    topology = json.load(f)

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

# ===== Generate HTML visualization =====
print("\nGenerating visualization...")

symbols = topology["symbols"]
geometry = topology["geometry"]
n_atoms = len(symbols)

# Build XYZ for 3Dmol.js
xyz_lines = [str(n_atoms), f"{METHOD}/{BASIS} water"]
for i in range(n_atoms):
    xyz_lines.append(f"{symbols[i]}  {geometry[3*i]:.6f}  {geometry[3*i+1]:.6f}  {geometry[3*i+2]:.6f}")
xyz_str = "\\n".join(xyz_lines)

energy_display = f"{total_energy:.10f}" if total_energy is not None else "N/A"
energy_kcal = f"{total_energy * 627.509474:.4f}" if total_energy is not None else "N/A"

dipole_html = ""
if dipole is not None and isinstance(dipole, list) and len(dipole) == 3:
    mag = (dipole[0]**2 + dipole[1]**2 + dipole[2]**2) ** 0.5
    dipole_html = f"""
        <div class="property">
            <span class="label">Dipole Moment</span>
            <span class="value">{mag:.4f} a.u.</span>
        </div>
        <div class="property small">
            <span class="label">Components (x, y, z)</span>
            <span class="value">[{dipole[0]:.4f}, {dipole[1]:.4f}, {dipole[2]:.4f}]</span>
        </div>
    """

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Single Point Energy — Water</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f8f9fa; color: #212529; }}
  .container {{ max-width: 900px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 8px; }}
  .subtitle {{ color: #6c757d; margin-bottom: 24px; }}
  .card {{ background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
           padding: 24px; margin-bottom: 20px; }}
  .energy-big {{ font-size: 2.2rem; font-weight: 700; color: #0d6efd; text-align: center;
                 padding: 16px 0; }}
  .energy-unit {{ font-size: 1rem; color: #6c757d; font-weight: 400; }}
  .properties {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .property {{ display: flex; justify-content: space-between; padding: 8px 12px;
               background: #f8f9fa; border-radius: 6px; }}
  .property .label {{ font-weight: 500; color: #495057; }}
  .property .value {{ font-family: 'SF Mono', Monaco, monospace; color: #212529; }}
  .property.small {{ font-size: 0.85rem; }}
  .viewer-container {{ width: 100%; height: 400px; border-radius: 8px; overflow: hidden; }}
  .method-badge {{ display: inline-block; background: #e9ecef; padding: 4px 10px;
                   border-radius: 4px; font-size: 0.85rem; font-family: monospace; }}
</style>
</head>
<body>
<div class="container">
  <h1>Single Point Energy — Water (H₂O)</h1>
  <p class="subtitle">
    <span class="method-badge">{METHOD}/{BASIS}</span>
  </p>

  <div class="card">
    <div class="energy-big">
      {energy_display} <span class="energy-unit">Hartree</span>
    </div>
    <div style="text-align:center; color:#6c757d; font-size:0.95rem;">
      {energy_kcal} kcal/mol
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:1.1rem; margin-bottom:12px;">Properties</h2>
    <div class="properties">
      <div class="property">
        <span class="label">Method</span>
        <span class="value">{METHOD}</span>
      </div>
      <div class="property">
        <span class="label">Basis Set</span>
        <span class="value">{BASIS}</span>
      </div>
      <div class="property">
        <span class="label">Atoms</span>
        <span class="value">{n_atoms}</span>
      </div>
      <div class="property">
        <span class="label">Formula</span>
        <span class="value">H₂O</span>
      </div>
      {dipole_html}
    </div>
  </div>

  <div class="card">
    <h2 style="font-size:1.1rem; margin-bottom:12px;">Molecular Structure</h2>
    <div id="viewer" class="viewer-container"></div>
  </div>
</div>

<script>
  var viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
  viewer.addModel("{xyz_str}", "xyz");
  viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{scale: 0.3}}}});
  viewer.zoomTo();
  viewer.render();
</script>
</body>
</html>
"""

html_path = OUTPUT_DIR / "spe_results.html"
html_path.write_text(html_content)
print(f"✓ Visualization saved: {html_path}")

# Also copy to docs static outputs
import shutil
docs_output = Path(__file__).parent / "../../docs/_static/outputs/spe_results.html"
docs_output.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(html_path, docs_output)
print(f"✓ Docs output saved: {docs_output}")

print("\n✓ All done! Open the HTML file in a browser to view results.")
