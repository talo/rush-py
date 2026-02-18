# Tutorial 1: Understanding Your Molecule's Charge Distribution — CHELPG Analysis in 2 Minutes

| | |
|---|---|
| **Time** | 2 minutes |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.10+, `rush-py` installed |

---

## What You'll Learn

- How to compute **CHELPG partial charges** for any molecule using Rush
- How to interpret charge distributions in a drug-discovery context

## The Problem You're Solving

When designing or evaluating drug candidates, you need to understand **where charge sits on a molecule**. Charge distribution drives:

- **Solubility** — polar regions interact with water
- **Binding affinity** — electrostatic complementarity with a protein target
- **Membrane permeability** — too much polarity and the molecule won't cross lipid bilayers
- **Reactivity** — highly charged atoms are sites for metabolism or chemical instability

CHELPG (CHarges from ELectrostatic Potentials using a Grid-based method) fits partial charges to reproduce the quantum-mechanical electrostatic potential on a grid around the molecule. The result is a set of atom-centered charges that represent the molecule's electrostatic "personality."

## Quick Start (5 lines)

```python
from pathlib import Path
from rush import exess
from rush.convert.pdb import from_pdb
import json

# 1. Load PDB file
pdb_content = Path("aspirin.pdb").read_text()
trc = from_pdb(pdb_content)

# 2. Convert to topology JSON
topology_json = trc.topology.to_json()
topology_json["schema_version"] = "0.2.0"
topology_path = Path("aspirin_topology.json")
topology_path.write_text(json.dumps(topology_json, indent=2))

# 3. Run CHELPG calculation (returns charges in ~30 seconds)
result = exess.chelpg(topology_path=topology_path, collect=True)
```

That's it! The `result` tuple contains your charges plus metadata.

### Get the PDB file

You can find the sample PDB file at [`examples/chelpg/data/aspirin.pdb`](https://github.com/talo/rush-py/blob/main/examples/chelpg/data/aspirin.pdb), or use any PDB file from the Protein Data Bank.

---

## Full Visualization & Analysis

For complete charge extraction, visualization (bar chart + interactive 3D), and interpretation, see the **full example script**:

👉 **[Complete CHELPG Example](https://github.com/talo/rush-py/tree/feat/exess-docs/examples/chelpg)**

This includes:
- ✅ HDF5 charge extraction
- ✅ Bar chart with RdBu coloring (red = positive, blue = negative)
- ✅ Interactive 3D structure visualization with charge-colored atoms
- ✅ Summary statistics (total charge, min/max per atom)

**Quick preview** (left: 3D molecule with charges, right: bar chart by atom):

<iframe src="./01-chelpg-preview.html" width="100%" height="700"></iframe>

---

## Why CHELPG Matters

For drug discovery, CHELPG charges tell you:

- **Where reactions happen** — highly positive atoms are electrophilic (good targets for nucleophiles)
- **Binding potential** — complementary charges drive protein interactions
- **Solubility & permeability** — charge distribution predicts bioavailability
- **Stability** — charged sites are vulnerable to metabolism

---

## Notes

- **Default parameters** — Uses RestrictedHF method with cc-pVDZ basis set (hardcoded in Rush API; no user override currently available)
- **Running time** — ~30 seconds cloud-side for a small molecule
- **Cost-effective** — CHELPG outsources QM to Rush infrastructure; you only pay for compute time

---

### Interactive CHELPG Output

Run the code above to reproduce this result, or explore the pre-computed output below:

<iframe src="../../_static/outputs/chelpg_aspirin.html" width="100%" height="600px" frameborder="0"></iframe>

---

## See Also

- [Rush Documentation](https://docs.rush.so)
- [CHELPG Method (Chemistry)](https://en.wikipedia.org/wiki/CHELPG)
- [Example Workflow](https://github.com/talo/rush-py/tree/feat/exess-docs/examples/chelpg)
