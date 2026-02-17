"""
Example: EXESS Geometry Optimization

This script demonstrates how to:
1. Run QM geometry optimization
2. Run ML (AIMNet) geometry optimization
3. Work with the optimization trajectory output

Tutorial: docs/tutorials/exess-optimization.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: benzene_t.json (provided in data/)
"""

import json
from pathlib import Path

from rush import Topology, exess
from rush.client import RunOpts, save_object

DATA_DIR = Path(__file__).parent / "data"
INPUT_FILE = DATA_DIR / "benzene_t.json"


# ===== Example 1: QM optimization =====
print("=" * 60)
print("Example 1: QM Geometry Optimization")
print("=" * 60)

# NOTE: Using RestrictedHF/STO-3G for demonstration purposes only.
# This is a very fast but low-accuracy method. For production results,
# use a higher-level method (e.g., RestrictedHF/cc-pVDZ or DFT).

out = exess.optimization(
    INPUT_FILE,
    100,  # Number of optimization iterations
    method="RestrictedHF",
    basis="STO-3G",
    standard_orientation="None",
    run_opts=RunOpts(name="Tutorial: Optimization using QM"),
    collect=True,
)


# ===== Example 2: ML optimization =====
print()
print("=" * 60)
print("Example 2: ML (AIMNet) Geometry Optimization")
print("=" * 60)

out = exess.optimization(
    INPUT_FILE,
    100,
    basis="STO-2G",
    optimization_keywords=exess.OptimizationKeywords(
        coordinate_system="Cartesian",
        algorithm="LBFGS",
        lbfgs_keywords=exess.LBFGSKeywords(),
    ),
    standard_orientation="None",
    qm_fragments=[],
    mm_fragments=[],
    run_opts=RunOpts(name="Tutorial: Optimization using ML"),
    collect=True,
)


# ===== Working with the output =====
print()
print("=" * 60)
print("Working with the optimization output")
print("=" * 60)

out_traj_path, out_info_path = [save_object(obj["path"]) for obj in out]
with open(out_traj_path) as f1, open(out_info_path) as f2:
    out_traj, out_info = [json.load(f) for f in (f1, f2)]

print("Num steps to convergence:", len(out_traj))

out_traj = [Topology.from_json(t) for t in out_traj]
print("First Atom's Coords")
print(f"  First step: {out_traj[0].geometry[:3]}")
print(f"  Final step: {out_traj[-1].geometry[:3]}")

# The below are only provided for QM regions
print("Final Step Info")
print(f"  Total energy: {out_info[-1]['total_energy']:.5f} Eh")
print(f"  Max gradient component: {out_info[-1]['max_gradient_component']:.2} Å")
