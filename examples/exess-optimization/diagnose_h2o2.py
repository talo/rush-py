"""Diagnose H2O2 optimization: check initial geometry, run optimization, compare dihedral angles."""
import json, math, sys
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "h2o2_t.json"

# 1. Check input geometry
with open(INPUT_FILE) as f:
    mol = json.load(f)

symbols = mol["symbols"]
geom = mol["geometry"]
n = len(symbols)
coords = np.array(geom).reshape(n, 3)

print("=" * 60)
print("INPUT GEOMETRY (h2o2_t.json)")
print("=" * 60)
for i, s in enumerate(symbols):
    print(f"  {s}: {coords[i]}")

# O-O distance
oo_dist = np.linalg.norm(coords[1] - coords[2])
print(f"\nO-O distance: {oo_dist:.4f} Å")

def dihedral(p1, p2, p3, p4):
    """Calculate dihedral angle in degrees."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return math.degrees(math.atan2(y, x))

d_initial = dihedral(coords[0], coords[1], coords[2], coords[3])
print(f"H-O-O-H dihedral angle: {d_initial:.2f}°")
print(f"All y-coords zero (planar)? {all(abs(coords[i,1]) < 1e-6 for i in range(n))}")

# 2. Run optimization
print("\n" + "=" * 60)
print("RUNNING OPTIMIZATION...")
print("=" * 60)

from rush import Topology, exess
from rush.client import RunOpts, save_object

out = exess.optimization(
    INPUT_FILE,
    100,
    method="RestrictedHF",
    basis="STO-3G",
    standard_orientation="None",
    run_opts=RunOpts(name="H2O2 dihedral diagnostic"),
    collect=True,
)

out_traj_path, out_info_path = [save_object(obj["path"]) for obj in out]
with open(out_traj_path) as f1, open(out_info_path) as f2:
    out_traj_raw, out_info = [json.load(f) for f in (f1, f2)]

print(f"Converged in {len(out_traj_raw)} steps")

# 3. Extract final geometry
final_topo = Topology.from_json(out_traj_raw[-1])
final_coords = np.array(final_topo.geometry).reshape(len(final_topo.symbols), 3)

print("\n" + "=" * 60)
print("FINAL GEOMETRY")
print("=" * 60)
for i, s in enumerate(final_topo.symbols):
    print(f"  {s}: {final_coords[i]}")

oo_dist_final = np.linalg.norm(final_coords[1] - final_coords[2])
print(f"\nO-O distance: {oo_dist_final:.4f} Å")

d_final = dihedral(final_coords[0], final_coords[1], final_coords[2], final_coords[3])
print(f"H-O-O-H dihedral angle: {d_final:.2f}°")

# 4. Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Initial dihedral: {d_initial:.2f}°")
print(f"  Final dihedral:   {d_final:.2f}°")
print(f"  Change:           {d_final - d_initial:.2f}°")
print(f"  Initial O-O:      {oo_dist:.4f} Å")
print(f"  Final O-O:        {oo_dist_final:.4f} Å")

energy_key = "total_energy" if "total_energy" in out_info[0] else "energy"
if energy_key in out_info[0]:
    e0 = out_info[0][energy_key]
    ef = out_info[-1][energy_key]
    print(f"  Initial energy:   {e0:.8f} Eh")
    print(f"  Final energy:     {ef:.8f} Eh")
    print(f"  Energy change:    {ef - e0:.8f} Eh")

if abs(d_final - d_initial) > 5:
    print("\n✅ Dihedral DID change! Problem is VISUALIZATION, not optimization.")
    print("   The 3Dmol viewer's zoomTo() auto-rotates both structures to look similar.")
    print("   Fix: Set a fixed camera orientation for both viewers.")
else:
    print("\n❌ Dihedral did NOT change significantly. Problem is the OPTIMIZER.")
