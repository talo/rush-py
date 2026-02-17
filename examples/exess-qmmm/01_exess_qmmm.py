"""
Example: EXESS QM/MM Simulation

This script demonstrates how to:
1. Run a basic QM/MM simulation
2. Build a minimal system manually (two water molecules)
3. Work with the simulation trajectory output

Tutorial: docs/tutorials/exess-qmmm.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input files: 6a5j_t.json, 6a5j_r.json (from tests/data/)
"""

import json
from itertools import batched
from pathlib import Path

from rush import Topology, exess
from rush.client import RunOpts, save_object
from rush.mol import Element, Fragment, Residue, Residues

DATA_DIR = Path(__file__).parent / "data"


# ===== Example 1: Basic QM/MM run =====
print("=" * 60)
print("Example 1: Basic QM/MM simulation")
print("=" * 60)

out = exess.qmmm(
    DATA_DIR / "6a5j_t.json",
    DATA_DIR / "6a5j_r.json",
    500,  # Number of timesteps
    qm_fragments=[6],  # QM region: just the fragment at index 6
    ml_fragments=[],    # ML region: empty, so rest is in the MM region
    run_opts=RunOpts(name="Tutorial: QM/MM"),
    collect=True,
)


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

with open("molecule_t.json", "w") as f_t:
    json.dump(topology.to_json(), f_t)
with open("molecule_r.json", "w") as f_r:
    json.dump(residues.to_json(), f_r)

out = exess.qmmm(
    topology_path="molecule_t.json",
    residues_path="molecule_r.json",
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
with open(out_file) as f:
    out_traj = json.load(f)["geometries"]

topology = Topology.from_json(Path("molecule_t.json"))
print("Atoms at First Step")
for atom_x, atom_y, atom_z in batched(topology.geometry, 3):
    print(f"  x: {atom_x:>7.4f}, y: {atom_y:>7.4f}, z: {atom_z:>7.4f}")

topology.geometry = out_traj[-1]
print("Atoms at Final Step")
for atom_x, atom_y, atom_z in batched(topology.geometry, 3):
    print(f"  x: {atom_x:>7.4f}, y: {atom_y:>7.4f}, z: {atom_z:>7.4f}")
