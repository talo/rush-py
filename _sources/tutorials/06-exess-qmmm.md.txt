# Tutorial 6: EXESS QM/MM Dynamics

**What you get:** A trajectory of atomic positions over time — a molecular movie where the critical region (your ligand, active site, or reaction center) is treated with quantum mechanics while the rest uses fast classical force fields.

| | |
|---|---|
| **Time** | ~5–30 minutes depending on system size and timesteps |
| **Skill level** | Intermediate |
| **Prerequisites** | Python 3.10+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Why This Matters

Static structures tell you *where* atoms are. Dynamics tells you *how they move* — which bonds vibrate, which water molecules exchange, how flexible a binding pocket is. Classical molecular dynamics (MD) handles this well for proteins, but it can't describe bond breaking, charge redistribution, or electronic effects in your region of interest.

**QM/MM** gives you the best of both worlds: quantum mechanics for the part that matters (a ligand, a catalytic site, a metal center) and molecular mechanics for everything else. EXESS extends this further with **Q4ML** — you can mix QM, MM, and ML (AIMNet neural network) regions in a single simulation.

---

## Quick Start: QM/MM on a Protein–Ligand System

```python
from rush import exess
from rush.client import RunOpts, save_object

out = exess.qmmm(
    "6a5j_t.json",    # Topology (atoms, coordinates, fragments)
    500,               # Number of MD timesteps
    "6a5j_r.json",    # Residues (residue names and assignments)
    qm_fragments=[6], # Treat fragment 6 with quantum mechanics
    ml_fragments=[],   # No ML region → everything else is MM
    run_opts=RunOpts(name="Tutorial: QM/MM"),
    collect=True,
)
```

This runs 500 timesteps of molecular dynamics where fragment 6 (e.g., a ligand or key residue) is computed with Hartree-Fock QM, and the remaining protein + solvent fragments use classical MM (OpenMM).

:::{admonition} Default settings
:class: note
Unless overridden, EXESS uses `method="RestrictedHF"`, `basis="STO-3G"`, and `temperature_kelvin=300.0`. The STO-3G basis is a minimal basis set — fast for testing, but not suitable for production dynamics. For meaningful simulations, use a larger basis like `cc-pVDZ`.
:::

---

## Minimal Example: Two Water Molecules from Scratch

To understand exactly what inputs QM/MM needs, let's build a system manually — two water molecules, all treated with QM:

```python
import json
from rush import exess
from rush.client import RunOpts
from rush import Topology, exess
from rush.mol import Element, Fragment, Residue, Residues

# Define two water molecules
topology = Topology(
    symbols=[Element.O, Element.H, Element.H, Element.O, Element.H, Element.H],
    geometry=[
         0.0000, 0.0000, 0.0000,   # Water 1: O
         0.7570, 0.5860, 0.0000,   #          H
        -0.7570, 0.5860, 0.0000,   #          H
         2.8000, 0.0000, 0.0000,   # Water 2: O
         3.5570, 0.5860, 0.0000,   #          H
         2.0430, 0.5860, 0.0000,   #          H
    ],
    fragments=[Fragment([0, 1, 2]), Fragment([3, 4, 5])],
)

residues = Residues(
    residues=[Residue([0, 1, 2]), Residue([3, 4, 5])],
    seqs=["HOH", "HOH"],
)

# Write input files
with open("molecule_t.json", "w") as f:
    json.dump(topology.to_json(), f)
with open("molecule_r.json", "w") as f:
    json.dump(residues.to_json(), f)

# Run all-QM dynamics
out = exess.qmmm(
    topology_path="molecule_t.json",
    residues_path="molecule_r.json",
    n_timesteps=100,
    trajectory=exess.Trajectory(include_waters=True),
    ml_fragments=[],
    mm_fragments=[],   # No ML or MM → everything is QM
    run_opts=RunOpts(name="Tutorial: QM/MM Water Dimer"),
    collect=True,
)
```

:::{admonition} Fragment assignment logic
:class: tip
EXESS requires that every fragment is assigned to exactly one of QM, MM, or ML. Setting `ml_fragments=[]` and `mm_fragments=[]` forces all fragments into the QM region. You can also explicitly set `qm_fragments=[0, 1]` for clarity.
:::

---

## Working with the Trajectory

The output is a JSON object containing `geometries` — a list of coordinate arrays, one per timestep. Each geometry is a flat list of floats (`[x1, y1, z1, x2, y2, z2, ...]`), matching the order of atoms in your Topology.

```python
import json
from itertools import batched
from pathlib import Path
from rush import Topology

out_file = save_object(out["path"])
with open(out_file) as f:
    geometries = json.load(f)["geometries"]

print(f"Trajectory has {len(geometries)} frames")

# Load the original topology and compare first/last frames
topology = Topology.from_json(Path("molecule_t.json"))

print("\nAtoms at First Step:")
for x, y, z in batched(topology.geometry, 3):
    print(f"  ({x:>7.4f}, {y:>7.4f}, {z:>7.4f})")

topology.geometry = geometries[-1]
print("\nAtoms at Final Step:")
for x, y, z in batched(topology.geometry, 3):
    print(f"  ({x:>7.4f}, {y:>7.4f}, {z:>7.4f})")
```

This example uses **6a5j**, a solution NMR structure of a small peptide. During the gas-phase NVE simulation, the peptide backbone flexes and side chains reorient. The surrounding MM environment would normally provide confining forces; in this gas-phase simulation, the structure is free to explore conformational space.

---

## Example Output

Explore the QM/MM trajectory visualization. Or run the code above to generate this yourself!

<iframe src="../_static/outputs/qmmm_results.html" width="100%" height="600px" frameborder="0"></iframe>

---

## Try It Yourself

The complete example is available in the rush-py repository:

👉 **[examples/exess-qmmm/](https://github.com/talo/rush-py/tree/main/examples/exess-qmmm){target="_blank"}**

You can download the files directly or clone the repository to run the full example.

---

## See Also

- {doc}`Optimization tutorial<04-exess-optimization>` — relax structures before running dynamics
- {doc}`Interaction energy tutorial<05-exess-interaction-energy>` — compute binding energies from trajectory snapshots
- {doc}`Exports tutorial<03-exess-exports>` — extract electronic properties during simulations
- [Full example script](https://github.com/talo/rush-py/tree/main/examples/exess-qmmm){target="_blank"}
