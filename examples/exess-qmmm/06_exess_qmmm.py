"""
Example: EXESS QM/MM Simulation

This script demonstrates how to:
1. Run a basic QM/MM simulation
2. Build a minimal system manually (two water molecules)
3. Work with the simulation trajectory output
4. Generate an interactive HTML visualization of the trajectory

Tutorial: docs/tutorials/exess-qmmm.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input files: 6a5j_t.json, 6a5j_r.json (provided in data/)

Output files (saved to qmmm-outputs/):
    - qmmm_results.html: Interactive visualization with animated 3D trajectory,
      QM/MM region highlighting, and summary statistics
"""

import json
from itertools import batched
from pathlib import Path

from rush import Topology, exess
from rush.client import RunOpts, save_object
from rush.mol import Element, Fragment, Residue, Residues

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "qmmm-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ===== Example 1: Basic QM/MM run =====
print("=" * 60)
print("Example 1: Basic QM/MM simulation")
print("=" * 60)

# ⚠️ TUTORIAL ONLY: STO-3G is a minimal basis set used here for speed/demonstration.
# It is NOT suitable for research or production use. For real work, use at least
# cc-pVDZ or larger (e.g., cc-pVTZ, aug-cc-pVDZ) with an appropriate method.

METHOD = "RestrictedHF"
BASIS = "STO-3G"
N_TIMESTEPS = 500
QM_FRAGMENTS = [6]
TEMPERATURE = 300  # Default temperature in Kelvin

topology_path = DATA_DIR / "6a5j_t.json"
residues_path = DATA_DIR / "6a5j_r.json"

out = exess.qmmm(
    topology_path,
    N_TIMESTEPS,
    residues_path,
    method=METHOD,
    basis=BASIS,
    qm_fragments=QM_FRAGMENTS,
    ml_fragments=[],
    run_opts=RunOpts(name="Tutorial: QM/MM"),
    collect=True,
)


# ===== Working with the output =====
print()
print("=" * 60)
print("Working with the QM/MM trajectory output")
print("=" * 60)

out_file = save_object(out["path"])
with open(out_file, encoding='utf-8') as f:
    out_data = json.load(f)

out_traj = out_data["geometries"]

# Load topology for atom info
with open(topology_path, encoding='utf-8') as f:
    topo_data = json.load(f)

symbols = topo_data["symbols"]
fragments = topo_data["fragments"]
n_atoms = len(symbols)

# Identify QM atom indices
qm_atom_indices = set()
for frag_idx in QM_FRAGMENTS:
    qm_atom_indices.update(fragments[frag_idx])

# MM fragment count = total fragments minus QM fragments
n_mm_fragments = len(fragments) - len(QM_FRAGMENTS)
n_qm_atoms = len(qm_atom_indices)
n_mm_atoms = n_atoms - n_qm_atoms

print(f"Total atoms: {n_atoms}")
print(f"QM atoms: {n_qm_atoms}, MM atoms: {n_mm_atoms}")
print(f"Trajectory frames: {len(out_traj)}")

# Print first/last frame info
initial_geom = out_traj[0] if out_traj else topo_data["geometry"]
final_geom = out_traj[-1] if out_traj else topo_data["geometry"]

print("First atom position:")
print(f"  Initial: ({initial_geom[0]:.4f}, {initial_geom[1]:.4f}, {initial_geom[2]:.4f})")
print(f"  Final:   ({final_geom[0]:.4f}, {final_geom[1]:.4f}, {final_geom[2]:.4f})")


# ===== Generate HTML Visualization =====
print()
print("=" * 60)
print("Generating HTML visualization")
print("=" * 60)


def geometry_to_xyz(syms, geom, frame_label=""):
    """Convert symbols + flat geometry list to XYZ format string."""
    n = len(syms)
    lines = [str(n), frame_label]
    for i in range(n):
        x, y, z = geom[3 * i], geom[3 * i + 1], geom[3 * i + 2]
        lines.append(f"{syms[i]}  {x:.6f}  {y:.6f}  {z:.6f}")
    return "\n".join(lines)


# Build all frames as XYZ strings
all_frames_xyz = []
for i, geom in enumerate(out_traj):
    all_frames_xyz.append(geometry_to_xyz(symbols, geom, f"Frame {i}"))

n_frames = len(all_frames_xyz)

# JSON-encode data for embedding in HTML
frames_js = json.dumps(all_frames_xyz)
qm_indices_js = json.dumps(sorted(qm_atom_indices))

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QM/MM Trajectory Results</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 28px 40px; }}
  .header h1 {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; }}
  .header p {{ opacity: 0.7; margin-top: 4px; font-size: 0.9rem; }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}

  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .stat-card {{ background: white; border-radius: 10px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .stat-card .label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 4px; }}
  .stat-card .value {{ font-size: 1.15rem; font-weight: 600; color: #1a1a2e; font-variant-numeric: tabular-nums; }}
  .stat-card .value.highlight {{ color: #e76f51; }}

  .panel {{ background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 24px; overflow: hidden; }}
  .panel-header {{ padding: 16px 20px; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 0.95rem; }}
  .panel-body {{ padding: 20px; }}

  .mol-viewer {{ width: 100%; height: 500px; border: 1px solid #e5e7eb; border-radius: 8px; position: relative; }}

  .controls {{ display: flex; align-items: center; gap: 12px; margin-top: 16px; flex-wrap: wrap; }}
  .controls button {{
    background: #1a1a2e; color: white; border: none; border-radius: 6px;
    padding: 8px 18px; font-size: 0.85rem; cursor: pointer; font-weight: 500;
    transition: background 0.15s;
  }}
  .controls button:hover {{ background: #0f3460; }}
  .controls button.active {{ background: #e76f51; }}
  .controls input[type="range"] {{ flex: 1; min-width: 200px; accent-color: #0f3460; }}
  .frame-label {{ font-size: 0.9rem; font-weight: 600; color: #1a1a2e; min-width: 120px; text-align: right; font-variant-numeric: tabular-nums; }}
  .speed-label {{ font-size: 0.8rem; color: #6b7280; }}

  .legend {{ display: flex; gap: 20px; margin-top: 12px; font-size: 0.85rem; color: #6b7280; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .legend-dot.qm {{ background: #e76f51; }}
  .legend-dot.mm {{ background: #8ecae6; }}
</style>
</head>
<body>
<div class="header">
  <h1>QM/MM Molecular Dynamics Trajectory</h1>
  <p>{METHOD}/{BASIS} &mdash; {n_frames} frames &mdash; {n_atoms} atoms ({n_qm_atoms} QM + {n_mm_atoms} MM)</p>
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
      <div class="label">Timesteps</div>
      <div class="value">{N_TIMESTEPS}</div>
    </div>
    <div class="stat-card">
      <div class="label">Temperature</div>
      <div class="value">{TEMPERATURE} K</div>
    </div>
    <div class="stat-card">
      <div class="label">QM Fragments</div>
      <div class="value highlight">{len(QM_FRAGMENTS)}</div>
    </div>
    <div class="stat-card">
      <div class="label">MM Fragments</div>
      <div class="value">{n_mm_fragments}</div>
    </div>
    <div class="stat-card">
      <div class="label">Total Atoms</div>
      <div class="value">{n_atoms}</div>
    </div>
  </div>

  <!-- 3D Trajectory Viewer -->
  <div class="panel">
    <div class="panel-header">Animated Trajectory Viewer</div>
    <div class="panel-body">
      <div id="viewer" class="mol-viewer"></div>
      <div class="controls">
        <button id="btn-play" onclick="togglePlay()">&#9654; Play</button>
        <button id="btn-step-back" onclick="stepFrame(-1)">&#9664;</button>
        <button id="btn-step-fwd" onclick="stepFrame(1)">&#9654;</button>
        <input type="range" id="scrubber" min="0" max="{n_frames - 1}" value="0" oninput="seekFrame(+this.value)">
        <span class="frame-label" id="frame-label">Frame 1 / {n_frames}</span>
      </div>
      <div class="controls">
        <span class="speed-label">Speed:</span>
        <button onclick="setSpeed(200)">0.5&times;</button>
        <button onclick="setSpeed(100)" class="active" id="speed-1x">1&times;</button>
        <button onclick="setSpeed(50)">2&times;</button>
        <button onclick="setSpeed(25)">4&times;</button>
      </div>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot qm"></div> QM region</div>
        <div class="legend-item"><div class="legend-dot mm"></div> MM region</div>
      </div>
    </div>
  </div>
</div>

<script>
const frames = {frames_js};
const qmIndices = new Set({qm_indices_js});
const nFrames = frames.length;
let currentFrame = 0;
let playing = false;
let interval = null;
let speed = 100; // ms per frame

const viewer = $3Dmol.createViewer('viewer', {{ backgroundColor: 'white' }});

function loadFrame(idx) {{
  currentFrame = idx;
  viewer.removeAllModels();
  viewer.addModel(frames[idx], 'xyz');

  // Style MM atoms (default)
  viewer.setStyle({{}}, {{
    stick: {{ radius: 0.1, color: '#8ecae6' }},
    sphere: {{ scale: 0.2, color: '#8ecae6' }}
  }});

  // Highlight QM atoms
  const qmSel = {{ index: Array.from(qmIndices) }};
  viewer.setStyle(qmSel, {{
    stick: {{ radius: 0.15, color: '#e76f51' }},
    sphere: {{ scale: 0.3, color: '#e76f51' }}
  }});

  viewer.render();
  document.getElementById('scrubber').value = idx;
  document.getElementById('frame-label').textContent = 'Frame ' + (idx + 1) + ' / ' + nFrames;
}}

function togglePlay() {{
  playing = !playing;
  const btn = document.getElementById('btn-play');
  if (playing) {{
    btn.innerHTML = '&#9646;&#9646; Pause';
    btn.classList.add('active');
    interval = setInterval(() => {{
      currentFrame = (currentFrame + 1) % nFrames;
      loadFrame(currentFrame);
    }}, speed);
  }} else {{
    btn.innerHTML = '&#9654; Play';
    btn.classList.remove('active');
    clearInterval(interval);
  }}
}}

function seekFrame(idx) {{
  loadFrame(+idx);
  if (playing) {{ clearInterval(interval); playing = false; togglePlay(); }}
}}

function stepFrame(delta) {{
  let next = (currentFrame + delta + nFrames) % nFrames;
  if (playing) {{ togglePlay(); }}
  loadFrame(next);
}}

function setSpeed(ms) {{
  speed = ms;
  document.querySelectorAll('.controls button[onclick^="setSpeed"]').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  if (playing) {{
    clearInterval(interval);
    interval = setInterval(() => {{
      currentFrame = (currentFrame + 1) % nFrames;
      loadFrame(currentFrame);
    }}, speed);
  }}
}}

// Load first frame
loadFrame(0);
viewer.zoomTo();
viewer.render();
</script>
</body>
</html>"""

html_path = OUTPUT_DIR / "qmmm_results.html"
html_path.write_text(html_content, encoding='utf-8')
print(f"✓ Visualization saved: {html_path}")
print(f"  Open in a browser to view the interactive trajectory.")


# ===== Example 2: Minimal QM/MM with manually-constructed water =====
print()
print("=" * 60)
print("Example 2: Minimal QM/MM (two water molecules)")
print("=" * 60)

topology = Topology(
    symbols=[Element.O, Element.H, Element.H, Element.O, Element.H, Element.H],
    geometry=[
         0.0000, 0.0000, 0.0000,
         0.7570, 0.5860, 0.0000,
        -0.7570, 0.5860, 0.0000,
         2.8000, 0.0000, 0.0000,
         3.5570, 0.5860, 0.0000,
         2.0430, 0.5860, 0.0000,
    ],
    fragments=[Fragment([0, 1, 2]), Fragment([3, 4, 5])],
)

residues = Residues(
    residues=[Residue([0, 1, 2]), Residue([3, 4, 5])],
    seqs=["HOH", "HOH"],
)

molecule_t_path = OUTPUT_DIR / "molecule_t.json"
molecule_r_path = OUTPUT_DIR / "molecule_r.json"

with open(molecule_t_path, "w", encoding='utf-8') as f_t:
    json.dump(topology.to_json(), f_t)
with open(molecule_r_path, "w", encoding='utf-8') as f_r:
    json.dump(residues.to_json(), f_r)

out = exess.qmmm(
    topology_path=molecule_t_path,
    residues_path=molecule_r_path,
    n_timesteps=100,
    trajectory=exess.Trajectory(include_waters=True),
    ml_fragments=[],
    mm_fragments=[],
    run_opts=RunOpts(name="Tutorial: QM/MM with Manually-Constructed Water"),
    collect=True,
)


# ===== Working with the output =====
print()
print("=" * 60)
print("Working with the QM/MM trajectory output")
print("=" * 60)

out_file = save_object(out["path"])
with open(out_file, encoding='utf-8') as f:
    out_traj = json.load(f)["geometries"]

topology = Topology.from_json(molecule_t_path)
print("Atoms at First Step")
for atom_x, atom_y, atom_z in batched(topology.geometry, 3):
    print(f"  x: {atom_x:>7.4f}, y: {atom_y:>7.4f}, z: {atom_z:>7.4f}")

topology.geometry = out_traj[-1]
print("Atoms at Final Step")
for atom_x, atom_y, atom_z in batched(topology.geometry, 3):
    print(f"  x: {atom_x:>7.4f}, y: {atom_y:>7.4f}, z: {atom_z:>7.4f}")
