# Tutorial 1: EXESS CHELPG Charge Analysis

| | |
|---|---|
| **Time** | 2 minutes |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.10+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

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

## Quick Start

```python
from pathlib import Path
import json

from rush import exess
from rush import from_pdb

# 1. Load PDB file
pdb_content = Path("aspirin.pdb").read_text()
trc = from_pdb(pdb_content)

# 2. Convert to topology JSON
topology_json = trc.topology.to_json()
topology_path = Path("aspirin_topology.json")
topology_path.write_text(json.dumps(topology_json, indent=2))

# 3. Run energy calculation with CHELPG export (returns charges in ~30 seconds)
result = exess.energy(
    topology_path=topology_path,
    frag_keywords=None,  # disable fragmentation for CHELPG
    export_keywords=exess.ExportKeywords(export_chelpg_charges=True),
).fetch()

# 4. Extract the charges
charges = result.exports["chelpg_charges"]
```

That's it! `charges` is a list of CHELPG partial charges, one per atom.

Using `run.fetch()` gives us the quickest access to inspecting the exported data in Python.

### Get the PDB file

You can find the sample PDB file at [`examples/exess-chelpg/data/aspirin.pdb`](https://github.com/talo/rush-py/blob/main/examples/exess-chelpg/data/aspirin.pdb){target="_blank"}, or use any PDB file from the Protein Data Bank.

---

## Full Visualization & Analysis

For complete charge extraction, visualization (bar chart + interactive 3D), and interpretation, see the **full example script**:

👉 **[Complete CHELPG Example](https://github.com/talo/rush-py/tree/main/examples/exess-chelpg){target="_blank"}**

This includes:
- ✅ Charge extraction from exports JSON
- ✅ Bar chart with RdBu coloring (red = negative/electron-rich, blue = positive/electron-poor)
- ✅ Interactive 3D structure visualization with charge-colored atoms
- ✅ Summary statistics (total charge, min/max per atom)

**Quick preview** (left: 3D molecule with charges, right: bar chart by atom):

<iframe src="../_static/outputs/chelpg_aspirin.html" width="100%" height="400"></iframe>

---

## Why CHELPG Matters

For drug discovery, CHELPG charges tell you:

- **Where reactions happen** — highly positive atoms are electrophilic (good targets for nucleophiles)
- **Binding potential** — complementary charges drive protein interactions
- **Solubility & permeability** — charge distribution predicts bioavailability
- **Stability** — charged sites are vulnerable to metabolism

---

## Notes

- **Default parameters** — Uses RestrictedKSDFT with a cc-pVDZ basis set unless you override them

---

## Try It Yourself

The complete example is available in the rush-py repository:

👉 **[examples/exess-chelpg/](https://github.com/talo/rush-py/tree/main/examples/exess-chelpg){target="_blank"}**

You can download the files directly or clone the repository to run the full example.

---

## See Also

- {doc}`EXESS Documentation <../exess/index>`
- [CHELPG Method (Chemistry)](https://en.wikipedia.org/wiki/CHELPG){target="_blank"}
- [Example Workflow](https://github.com/talo/rush-py/tree/main/examples/exess-chelpg){target="_blank"}
