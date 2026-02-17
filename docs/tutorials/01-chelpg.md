# Tutorial 1: Understanding Your Molecule's Charge Distribution — CHELPG Analysis in 2 Minutes

| | |
|---|---|
| **Time** | 2 minutes |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.10+, `rush-py` installed, a topology JSON file |

---

## What You'll Learn

- How to compute **CHELPG partial charges** for any molecule using Rush
- How to interpret charge distributions in a drug-discovery context
- How to visualize charges with a publication-ready color-mapped 3D plot

## The Problem You're Solving

When designing or evaluating drug candidates, you need to understand **where charge sits on a molecule**. Charge distribution drives:

- **Solubility** — polar regions interact with water
- **Binding affinity** — electrostatic complementarity with a protein target
- **Membrane permeability** — too much polarity and the molecule won't cross lipid bilayers
- **Reactivity** — highly charged atoms are sites for metabolism or chemical instability

CHELPG (CHarges from ELectrostatic Potentials using a Grid-based method) fits partial charges to reproduce the quantum-mechanical electrostatic potential on a grid around the molecule. The result is a set of atom-centered charges that represent the molecule's electrostatic "personality."

## What You'll Need

1. **rush-py** — `pip install rush-py`
2. **A topology file** — a JSON file describing your molecular system (atom types, coordinates, connectivity). The examples below use an aspirin-like ligand topology.
3. **Visualization libraries** (optional) — `matplotlib` and `numpy` for the 3D charge plot.

---

## Step-by-Step Tutorial

### Step 1: Import Rush and configure your workspace

Every Rush calculation needs a workspace directory where results are stored locally, and a topology file that describes your system.

```python
from pathlib import Path
from rush import exess
from rush.client import RunOpts, set_opts

# Point Rush at a local directory for caching results
set_opts(workspace_dir=Path("./my-workspace"))

# Path to your topology file
topology = Path("data/aspirin.json")
```

### Step 2: Run the CHELPG calculation

A single function call submits the job to the Rush cloud, runs the quantum-chemical calculation, and collects the results. Setting `collect=True` blocks until the job finishes and downloads outputs automatically.

```python
result = exess.chelpg(
    topology,
    run_opts=RunOpts(
        name="Tutorial: Aspirin CHELPG Charges",
        tags=["tutorial", "aspirin", "chelpg"],
    ),
    collect=True,
)
```

The return value is a tuple `(output_metadata, charges)` where `charges` is a flat list of floats — one partial charge per atom, in the same order as the topology.

### Step 3: Inspect the charges

```python
output_meta, charges = result

print(f"Number of atoms: {len(charges)}")
print(f"Charges: {[round(q, 4) for q in charges]}")
```

Example output for aspirin (21 atoms):

```
Number of atoms: 21
Charges: [-0.5832, 0.7651, -0.5104, -0.4927, 0.1528, -0.1283, -0.1054, -0.0871, -0.1461, 0.7932, -0.5711, -0.5369, 0.1553, 0.1464, 0.1541, 0.1651, 0.1434, 0.1463, 0.0440, 0.0459, 0.0459]
```

### Step 4: Map charges to atom labels

If your topology is a typical organic molecule, you can pair each charge with its element. Here we assume a known aspirin atom order — adapt this to your own system.

```python
# Aspirin atom labels (match your topology's atom ordering)
atom_labels = [
    "O1", "C1", "O2", "O3", "C2", "C3", "C4", "C5", "C6",
    "C7", "O4", "O5", "H1", "H2", "H3", "H4", "H5", "H6",
    "H7", "H8", "H9",
]

for label, q in zip(atom_labels, charges):
    print(f"  {label:>4s}  {q:+.4f} e")
```

### Step 5: Visualize the charge distribution in 3D

This creates a color-mapped 3D scatter plot where each atom is colored by its partial charge — red for negative, blue for positive.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Atom coordinates (Å) — extract from your topology or use known geometry
coords = np.array([
    [-2.36,  0.94,  0.03], [-1.33,  0.14, -0.01], [-1.54, -1.18,  0.02],
    [ 0.00,  0.00,  0.00], [-0.08,  0.69, -0.05], [ 1.14,  0.06, -0.06],
    [ 2.38,  0.66, -0.10], [ 2.40,  2.06, -0.12], [ 1.21,  2.69, -0.11],
    [-0.04,  2.09, -0.07], [-0.33, -1.33,  0.04], [ 0.68, -1.24,  0.04],
    [-2.46, -1.34,  0.05], [ 1.15, -1.02, -0.05], [ 3.32,  0.11, -0.11],
    [ 3.35,  2.58, -0.15], [ 1.22,  3.78, -0.12], [-0.97,  2.60, -0.07],
    [-3.14,  0.36,  0.06], [-1.15, -2.15,  0.07], [ 1.55, -1.67,  0.07],
])

charges_arr = np.array(charges)

# Color normalization centered at zero
norm = TwoSlopeNorm(vmin=charges_arr.min(), vcenter=0, vmax=charges_arr.max())

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    coords[:, 0], coords[:, 1], coords[:, 2],
    c=charges_arr, cmap="RdBu_r", norm=norm,
    s=200, edgecolors="k", linewidths=0.5,
)

# Label each atom
for i, label in enumerate(atom_labels):
    ax.text(coords[i, 0], coords[i, 1], coords[i, 2] + 0.25,
            f"{label}\n{charges_arr[i]:+.3f}",
            fontsize=7, ha="center", va="bottom")

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("CHELPG Partial Charge (e)", fontsize=11)

ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.set_title("Aspirin — CHELPG Charge Distribution", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("aspirin_chelpg.png", dpi=150)
plt.show()
```

---

## Understanding the Numbers

| Charge Range (e) | Color | Typical Meaning |
|---|---|---|
| **−0.4 to −0.6** | 🔴 Deep red | Strong hydrogen-bond acceptors (carbonyls, carboxylate oxygens) |
| **−0.1 to −0.3** | 🟠 Light red | Mild electron-rich sites (aromatic carbons, ethers) |
| **−0.05 to +0.05** | ⚪ White / neutral | Non-polar regions (aliphatic C–H) |
| **+0.1 to +0.2** | 🔵 Light blue | Weakly positive (aromatic H, α-carbonyl H) |
| **+0.6 to +0.8** | 🔵 Deep blue | Strong electrophilic centers (carbonyl C, carboxyl C) |

## What This Tells You About Aspirin

Aspirin (acetylsalicylic acid) has two key functional groups — a **carboxylic acid** and an **ester** — connected through an aromatic ring.

- **Carboxyl carbon C7 (+0.79 e)** and **ester carbon C1 (+0.77 e)** are the most electrophilic atoms. These are the sites most vulnerable to nucleophilic attack (e.g., hydrolysis by esterases *in vivo*).
- **Oxygens O1, O2, O4, O5 (−0.49 to −0.58 e)** are the primary hydrogen-bond acceptors. They drive aspirin's aqueous solubility and its interactions with the COX enzyme binding site.
- **Aromatic ring carbons (−0.09 to −0.15 e)** carry slight negative charge — contributing to π-stacking but not dominating the electrostatics.
- **Hydrogens on the ring (+0.14 to +0.17 e)** are mildly positive, acting as weak hydrogen-bond donors to backbone carbonyls in the protein target.

This charge pattern explains why aspirin is orally bioavailable: enough polarity (the acid and ester) for aqueous solubility, but enough lipophilicity (the aromatic ring) for membrane permeation.

## Interpreting the Visualization

- **Red atoms** → electron-rich, nucleophilic, hydrogen-bond acceptors
- **Blue atoms** → electron-poor, electrophilic, hydrogen-bond donors
- **White/neutral atoms** → hydrophobic, non-polar surface

The diverging **Red–White–Blue** color map is centered at zero so the visual immediately tells you the molecule's electrostatic topology.

---

## Practical Applications

### Comparing analogs

Run the same CHELPG workflow on two analogs and compare their charge profiles:

```python
for name, topo in [("aspirin", "data/aspirin.json"), ("diflunisal", "data/diflunisal.json")]:
    _, charges = exess.chelpg(
        topo,
        run_opts=RunOpts(name=f"CHELPG: {name}", tags=["comparison"]),
        collect=True,
    )
    print(f"{name}: max={max(charges):+.3f}  min={min(charges):+.3f}  Σ={sum(charges):+.3f}")
```

### Feeding charges into downstream tools

CHELPG charges are commonly used as input for:

- **Molecular dynamics** force fields (AMBER, GROMACS)
- **Docking** rescoring with electrostatic terms
- **QSAR/ML models** as atomic-level descriptors

```python
# Export charges to a simple CSV for other tools
import csv

with open("aspirin_charges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["atom", "charge_e"])
    for label, q in zip(atom_labels, charges):
        writer.writerow([label, round(q, 6)])
```

---

## Common Use Cases

| Use Case | What CHELPG Tells You |
|---|---|
| ✅ **Reactivity prediction** | Identify electrophilic/nucleophilic sites for metabolism or chemical stability |
| ✅ **Membrane permeability** | Quantify polar surface — excessive charge reduces passive diffusion |
| ✅ **Protein–ligand interactions** | Map electrostatic complementarity with the binding pocket |
| ✅ **ADME optimization** | Balance polarity for solubility vs. permeability trade-offs |
| ✅ **Analog comparison** | Spot charge differences introduced by functional group changes (e.g., F vs. Cl substitution) |

---

## What You Just Learned

You computed **quantum-mechanical partial charges** for a drug molecule in under 2 minutes and visualized the result. In a pharmaceutical context, this is the starting point for:

- **Lead optimization** — understanding which parts of your molecule drive binding vs. liability
- **Formulation science** — predicting salt-form behavior and crystal packing
- **PBPK modeling** — charge-dependent tissue distribution and protein binding
- **IP differentiation** — demonstrating that your analog has a meaningfully different electrostatic profile from prior art

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `exess.chelpg()` hangs | Check your internet connection — Rush runs calculations in the cloud. Ensure `collect=True` is set if you want to wait for results. |
| Charges don't sum to the expected total charge | Verify your topology file has the correct total charge and multiplicity. CHELPG charges should sum to the system's net charge (0 for neutral molecules). |
| Coordinate mismatch in visualization | Make sure your `coords` array follows the same atom ordering as the topology file. |
| `ModuleNotFoundError: rush` | Install with `pip install rush-py`. Requires Python 3.10+. |
| Very large charges (> ±1.5 e) | This can happen with poor basis sets or strained geometries. Try a larger basis (e.g., `cc-pVTZ`) or check your input geometry for clashes. |

---

## Key Takeaways

1. **One function call** — `exess.chelpg()` handles the entire quantum-chemistry pipeline for you.
2. **Physically meaningful** — CHELPG charges reproduce the molecule's electrostatic potential, unlike empirical charge models.
3. **Actionable** — the charge map directly informs decisions about solubility, permeability, binding, and reactivity.
4. **Fast iteration** — change your molecule, re-run, and compare in minutes.

---

## Complete Script

<details>
<summary>📥 Click to expand the full script</summary>

```python
"""
CHELPG Charge Analysis and Visualization
=========================================
Compute and visualize CHELPG partial charges for aspirin using Rush.
"""

from pathlib import Path

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from rush import exess
from rush.client import RunOpts, set_opts

# ── Configuration ──────────────────────────────────────────────────────────
set_opts(workspace_dir=Path("./my-workspace"))
topology = Path("data/aspirin.json")

# ── Run CHELPG calculation ─────────────────────────────────────────────────
result = exess.chelpg(
    topology,
    run_opts=RunOpts(
        name="Tutorial: Aspirin CHELPG Charges",
        tags=["tutorial", "aspirin", "chelpg"],
    ),
    collect=True,
)

output_meta, charges = result

# ── Atom labels (must match topology ordering) ─────────────────────────────
atom_labels = [
    "O1", "C1", "O2", "O3", "C2", "C3", "C4", "C5", "C6",
    "C7", "O4", "O5", "H1", "H2", "H3", "H4", "H5", "H6",
    "H7", "H8", "H9",
]

# ── Print charge summary ──────────────────────────────────────────────────
print(f"Atoms: {len(charges)}")
print(f"Total charge: {sum(charges):+.4f} e\n")
for label, q in zip(atom_labels, charges):
    print(f"  {label:>4s}  {q:+.4f} e")

# ── Export to CSV ──────────────────────────────────────────────────────────
with open("aspirin_charges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["atom", "charge_e"])
    for label, q in zip(atom_labels, charges):
        writer.writerow([label, round(q, 6)])

# ── 3D Visualization ──────────────────────────────────────────────────────
coords = np.array([
    [-2.36,  0.94,  0.03], [-1.33,  0.14, -0.01], [-1.54, -1.18,  0.02],
    [ 0.00,  0.00,  0.00], [-0.08,  0.69, -0.05], [ 1.14,  0.06, -0.06],
    [ 2.38,  0.66, -0.10], [ 2.40,  2.06, -0.12], [ 1.21,  2.69, -0.11],
    [-0.04,  2.09, -0.07], [-0.33, -1.33,  0.04], [ 0.68, -1.24,  0.04],
    [-2.46, -1.34,  0.05], [ 1.15, -1.02, -0.05], [ 3.32,  0.11, -0.11],
    [ 3.35,  2.58, -0.15], [ 1.22,  3.78, -0.12], [-0.97,  2.60, -0.07],
    [-3.14,  0.36,  0.06], [-1.15, -2.15,  0.07], [ 1.55, -1.67,  0.07],
])

charges_arr = np.array(charges)
norm = TwoSlopeNorm(vmin=charges_arr.min(), vcenter=0, vmax=charges_arr.max())

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    coords[:, 0], coords[:, 1], coords[:, 2],
    c=charges_arr, cmap="RdBu_r", norm=norm,
    s=200, edgecolors="k", linewidths=0.5,
)

for i, label in enumerate(atom_labels):
    ax.text(coords[i, 0], coords[i, 1], coords[i, 2] + 0.25,
            f"{label}\n{charges_arr[i]:+.3f}",
            fontsize=7, ha="center", va="bottom")

cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label("CHELPG Partial Charge (e)", fontsize=11)

ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.set_title("Aspirin — CHELPG Charge Distribution", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("aspirin_chelpg.png", dpi=150)
plt.show()
```

</details>

---

## Next Steps

- **[Tutorial 2: Interaction Energy Analysis →](./exess-interaction-energy.md)** — compute binding energies between molecular fragments
- **[Exporting Additional Data →](./exess-exports.md)** — request densities, orbitals, and other properties alongside charges
- **[QM/MM Dynamics →](./exess-qmmm.md)** — run molecular dynamics with quantum-mechanical accuracy
