"""
Example: EXESS Geometry Optimization

This script demonstrates how to:
1. Run QM geometry optimization
2. Work with the optimization trajectory output
3. Generate an interactive HTML visualization of results

Tutorial: https://exess.qdx.co/docs/tutorials/04-exess-optimization.html

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: ethene_twisted_t.json (provided in data/) - twisted ethene optimizes to planar, showing π-bond constraint

Output files (saved to optimization-outputs/):
    - optimization_results.html: Interactive visualization with energy plot,
      3D structures, and summary statistics
"""

import json
from pathlib import Path

from rush import exess
from rush.client import RunOpts

DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "ethene_twisted_t.json"
OUTPUT_DIR = Path(__file__).parent / "optimization-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ===== Example 1: QM optimization =====
print("=" * 60)
print("Example 1: QM Geometry Optimization")
print("=" * 60)

# ⚠️ TUTORIAL ONLY: STO-3G is a minimal basis set used here for speed/demonstration.
# It is NOT suitable for research or production use. For real work, use at least
# cc-pVDZ or larger (e.g., cc-pVTZ, aug-cc-pVDZ) with an appropriate method.

METHOD = "RestrictedHF"
BASIS = "STO-3G"

run = exess.optimization(
    INPUT_FILE,
    100,  # Number of optimization iterations
    method=METHOD,
    basis=BASIS,
    standard_orientation="None",
    run_opts=RunOpts(name="Tutorial: Optimization using QM"),
)


# ===== Working with the output =====
print()
print("=" * 60)
print("Working with the optimization output")
print("=" * 60)

paths = run.save()
print(f"Saved files: {paths}")

res = run.fetch()

out_traj = res.trajectory
out_info = res.steps

print("Num steps to convergence:", len(out_traj))
print("First Atom's Coords")
print(f"  First step: {out_traj[0].geometry[:3]}")
print(f"  Final step: {out_traj[-1].geometry[:3]}")

# The below are only provided for QM regions
print("Final Step Info")
if out_info[-1].total_energy is not None:
    print(f"  Total energy: {out_info[-1].total_energy:.5f} Eh")
else:
    print("  Energy: (not available in output)")
if out_info[-1].max_gradient_component is not None:
    print(f"  Max gradient component: {out_info[-1].max_gradient_component:.2} Å")
else:
    print("  Max gradient component: (not available in output)")


# ===== Generate HTML Visualization =====
print()
print("=" * 60)
print("Generating HTML visualization")
print("=" * 60)


def topology_to_xyz(topo):
    """Convert a Topology object to XYZ format string."""
    symbols = topo.symbols
    geom = topo.geometry
    n_atoms = len(symbols)
    lines = [str(n_atoms), ""]
    for i in range(n_atoms):
        x, y, z = geom[3 * i], geom[3 * i + 1], geom[3 * i + 2]
        lines.append(f"{symbols[i]}  {x:.6f}  {y:.6f}  {z:.6f}")
    return "\n".join(lines)


initial_xyz = topology_to_xyz(out_traj[0])
final_xyz = topology_to_xyz(out_traj[-1])

if out_info[0].total_energy is None:
    raise KeyError(
        "Cannot find total_energy in optimization output. "
        f"Available values: {out_info[0]!r}"
    )
energies = [step.total_energy for step in out_info]
steps = list(range(len(energies)))
initial_energy = energies[0]
final_energy = energies[-1]
energy_change = final_energy - initial_energy
final_max_grad = out_info[-1].max_gradient_component or "N/A"
n_steps = len(out_traj)

# Escape XYZ strings for JavaScript embedding
initial_xyz_js = json.dumps(initial_xyz)
final_xyz_js = json.dumps(final_xyz)
energies_js = json.dumps(energies)
steps_js = json.dumps(steps)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Geometry Optimization Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 28px 40px; }}
  .header h1 {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; }}
  .header p {{ opacity: 0.7; margin-top: 4px; font-size: 0.9rem; }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}

  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: white; border-radius: 10px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .stat-card .label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 4px; }}
  .stat-card .value {{ font-size: 1.15rem; font-weight: 600; color: #1a1a2e; font-variant-numeric: tabular-nums; }}
  .stat-card .value.positive {{ color: #059669; }}
  .stat-card .value.negative {{ color: #dc2626; }}

  .panel {{ background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 24px; overflow: hidden; }}
  .panel-header {{ padding: 16px 20px; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 0.95rem; }}
  .panel-body {{ padding: 20px; }}

  .viewers {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .viewer-col {{ text-align: center; }}
  .viewer-col h3 {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.04em; }}
  .mol-viewer {{ width: 100%; height: 380px; border: 1px solid #e5e7eb; border-radius: 8px; position: relative; }}

  canvas {{ max-width: 100%; }}
</style>
</head>
<body>
<div class="header">
  <h1>Geometry Optimization Results</h1>
  <p>{METHOD}/{BASIS} &mdash; {n_steps} optimization steps</p>
</div>
<div class="container">

  <!-- Summary Stats -->
  <div class="summary">
    <div class="stat-card">
      <div class="label">Method</div>
      <div class="value">{METHOD}</div>
    </div>
    <div class="stat-card">
      <div class="label">Basis Set</div>
      <div class="value">{BASIS}</div>
    </div>
    <div class="stat-card">
      <div class="label">Steps</div>
      <div class="value">{n_steps}</div>
    </div>
    <div class="stat-card">
      <div class="label">Initial Energy</div>
      <div class="value">{initial_energy:.8f} Eh</div>
    </div>
    <div class="stat-card">
      <div class="label">Final Energy</div>
      <div class="value">{final_energy:.8f} Eh</div>
    </div>
    <div class="stat-card">
      <div class="label">Energy Change</div>
      <div class="value {"positive" if energy_change < 0 else "negative"}">{energy_change:.8f} Eh</div>
    </div>
    <div class="stat-card">
      <div class="label">Final Max Gradient</div>
      <div class="value">{final_max_grad:.2e} &#x212B;</div>
    </div>
  </div>

  <!-- Energy Convergence -->
  <div class="panel">
    <div class="panel-header">Energy Convergence</div>
    <div class="panel-body">
      <canvas id="energyChart" height="90"></canvas>
    </div>
  </div>

  <!-- 3D Structures -->
  <div class="panel">
    <div class="panel-header">Molecular Structure: Before &amp; After</div>
    <div class="panel-body">
      <div class="viewers">
        <div class="viewer-col">
          <h3>Initial Geometry (Step 0)</h3>
          <div id="viewer-initial" class="mol-viewer"></div>
        </div>
        <div class="viewer-col">
          <h3>Optimized Geometry (Step {n_steps - 1})</h3>
          <div id="viewer-final" class="mol-viewer"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// Energy convergence chart
const ctx = document.getElementById('energyChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {steps_js},
    datasets: [{{
      label: 'Total Energy (Eh)',
      data: {energies_js},
      borderColor: '#0f3460',
      backgroundColor: 'rgba(15, 52, 96, 0.08)',
      fill: true,
      tension: 0.3,
      pointRadius: 3,
      pointBackgroundColor: '#0f3460',
      borderWidth: 2
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => 'Energy: ' + ctx.parsed.y.toFixed(8) + ' Eh' }} }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Optimization Step', font: {{ weight: 'bold' }} }} }},
      y: {{ title: {{ display: true, text: 'Total Energy (Eh)', font: {{ weight: 'bold' }} }} }}
    }}
  }}
}});

// 3D molecular viewers
function setupViewer(elementId, xyzData) {{
  let viewer = $3Dmol.createViewer(elementId, {{ backgroundColor: 'white' }});
  viewer.addModel(xyzData, 'xyz');
  viewer.setStyle({{}}, {{ stick: {{ radius: 0.12, colorscheme: 'Jmol' }}, sphere: {{ scale: 0.25, colorscheme: 'Jmol' }} }});
  viewer.zoomTo();
  // Lock camera so both viewers show the same angle (45° view)
  viewer.rotate(45, 'x');
  viewer.rotate(30, 'y');
  viewer.render();
  return viewer;
}}

setupViewer('viewer-initial', {initial_xyz_js});
setupViewer('viewer-final', {final_xyz_js});
</script>
</body>
</html>"""

html_path = OUTPUT_DIR / "optimization_results.html"
html_path.write_text(html_content, encoding="utf-8")
print(f"✓ Visualization saved: {html_path}")
print("  Open in a browser to view interactive results.")
