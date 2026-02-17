<<<<<<< HEAD
# EXESS Interaction Energy

This tutorial walks through interaction energy calculations with EXESS using **rush-py**.

:::{note}
You still need `RUSH_TOKEN` and `RUSH_PROJECT` in your environment, but as of February 1, 2026, they are only required on first use (not at import time).
:::

## Prerequisites

Set your Rush environment variables before running the client:

- `RUSH_TOKEN`
- `RUSH_PROJECT`
- `RUSH_ENDPOINT` (optional)

All examples below reference test fixtures in [`tests/data/`](https://github.com/talo/rush-py/tree/main/tests/data).

## Common setup

```{code-block} python
:caption: setup.py

from pathlib import Path

from rush import exess
from rush.client import RunOpts, set_opts

set_opts(workspace_dir=Path.cwd() / "tutorial-runs")
DATA_DIR = Path.cwd() / "tests" / "data"
```

## Example: fragment-based interaction energy

This example follows [`tests/test_exess_interaction_energy.py`](https://github.com/talo/rush-py/blob/main/tests/test_exess_interaction_energy.py). It computes interaction energy between a ligand fragment and the rest of the system, using nearby fragments for the interaction region.

```{code-block} python
:caption: interaction_energy.py

from rush import Topology, exess
from rush.client import RunOpts

lig_idx = 93

topology = Topology.from_json(DATA_DIR / "tyk2_ejm_31_t.json")
frag_idcs = topology.get_fragments_near_fragment(lig_idx, 6.0) + [lig_idx]

res = exess.interaction_energy(
    DATA_DIR / "tyk2_ejm_31_t.json",
    lig_idx,
    frag_keywords=exess.FragKeywords(
        level="Trimer",
        dimer_cutoff=5.0,
        trimer_cutoff=1.0,
        cutoff_type="Centroid",
        distance_metric="Min",
        included_fragments=frag_idcs,
    ),
    run_opts=RunOpts(
        name="Tutorial: EXESS Interaction Energy",
        tags=["rush-py", "tutorial", "interaction-energy"],
=======
# EXESS: Interaction Energy

EXESS can leverage its fragmentation capabilities to calculate an interaction energy between a single fragment and the rest of the system. In a receptor-ligand complex, we can place the ligand in its own fragment and use it as the reference fragment. This doesn't constitute a binding free energy between the two, but does provide some information about the magnitude of the energy involved.

:::{admonition} Reminder
:class: attention
Don't forget to set `RUSH_TOKEN` and `RUSH_PROJECT`!
:::

## Fragment-based interaction energy

First, let's do a basic interaction energy calculation between a ligand fragment and the rest of the system. You can find the QDX Topology file used in the rush-py repo's [`tests/data/`](https://github.com/talo/rush-py/tree/main/tests/data) folder.
```{code-block} python
:caption: interaction_energy.py
import json

from rush import exess
from rush.client import RunOpts

out = exess.interaction_energy(
    "tyk2_ejm_31_t.json",
    93,  # This is the index of the fragment that contains the ligand 
    frag_keywords=exess.FragKeywords(
        level="Trimer",
        dimer_cutoff=10.0,
        trimer_cutoff=5.0,
        cutoff_type="Centroid",
        distance_metric="Min",
    ),
    run_opts=RunOpts(
        name="Tutorial: Interaction Energy Basic",
>>>>>>> 01791af6000f243f0b54f3fbd35c775cd99b46f7
    ),
    collect=True,
)

<<<<<<< HEAD
exess.save_energy_outputs(res)
```

**Example output shape** (object-store paths):

```{code-block} python
[
    {"path": "<uuid>", "size": 0, "format": "json"},
    {"path": "<uuid>", "size": 0, "format": "bin"},
]
```

:::{tip}
Increase the pocket cutoff (e.g., 8 to 12 Angstrom) if you need a broader interaction region. The `included_fragments` list controls which fragments are considered in the interaction energy calculation.
:::

## Optional: prepare a complex from PDB (from `demos/`)

The demo `demos/jnj_prep+interaction.py` shows a preparation pipeline using a PDB and ligand residue names. Below is a cleaned-up version that uses the current `rush` imports and feeds directly into `interaction_energy`.

```{code-block} python
:caption: prepare_and_interaction.py

from pathlib import Path

from rush import to_json, exess
from rush.client import RunOpts
from rush.prepare_complex import prepare_complex

pdb_path = Path("data/1hsg.pdb")
ligand_names = ["MK1", "HOH"]

trc = prepare_complex(
    pdb_path,
    ligand_names,
    run_opts=RunOpts(name="Tutorial: Prepare Complex"),
    collect=True,
)

# Save TRC to JSON so it can be used by EXESS
trc_path = Path("1hsg_complex.json")
trc_path.write_text(to_json(trc))

# Find ligand fragment index + nearby pocket
lig_idx = trc.residues.seqs.index("MK1")
frag_idcs = trc.topology.get_fragments_near_fragment(lig_idx, 5.0) + [lig_idx]

res = exess.interaction_energy(
    trc_path,
    lig_idx,
    frag_keywords=exess.FragKeywords(
        level="Trimer",
        dimer_cutoff=25.0,
        trimer_cutoff=10.0,
        cutoff_type="Centroid",
        distance_metric="Min",
        included_fragments=frag_idcs,
    ),
    run_opts=RunOpts(name="Tutorial: Interaction Energy (Prepared Complex)"),
    collect=True,
)

exess.save_energy_outputs(res)
```

:::{admonition} Notes
:class: note
- `prepare_complex` requires RDKit and uses the same workflow as the demo.
- If you see a `RunError`, check the Rush UI for details or increase `run_spec` resources.
:::
=======
out_file, _ = exess.save_energy_outputs(out)
with open(out_file) as f:
    out_data = json.load(f)
print(f"Interaction energy: {out_data['qmmbe']['expanded_hf_energy']}")
```

:::{admonition} The other, ignored output
:class: note
The second output of the module, the one being ignored as `_` from the return value of `save_energy_outputs(out)`, contains additional data that EXESS only calculates and exports upon request. See the {doc}`Exports tutorial<exess-exports>` for more info.
:::

## End-to-end interaction energy calculation using a complex from the PDB

### Step 1: Prepare the system

Before submitting a system as input to EXESS, we have to ensure it has the correct protonation state, which in this context means that it:
- must contain all its hydrogens, and
- that each fragment must be annotated with its formal charge.

Rush contains a module called Prepare Complex that does this for a system with at least one protein and ligand in it. The ligands are identified via their residue names (PDB) or residue labels (QDX's TRC), which correspond to each other and share the same data when converting between the formats.
```{code-block} python
import json
from pathlib import Path

from rush.client import RunOpts
from rush.prepare_complex import prepare_complex

trc = prepare_complex(
    Path(__file__).parent.parent / "demos/data/1hsg.pdb",
    ligand_names=["MK1", "HOH"],
    run_opts=RunOpts(name="Tutorial: Interaction Energy E2E - Prepare Complex"),
    collect=True,
)

# Print the charged amino acids
print("Charged amino acids:")
for i, (res_name, formal_charge) in enumerate(
    zip(trc.residues.seqs, trc.topology.fragment_formal_charges)
):
    if int(formal_charge) != 0:
        print(f"{i:>4} {res_name}: {int(formal_charge):+}")
```

:::{admonition} Libraries used
:class: note
The Prepare Complex module uses PDBFixer and PDB2PQR to fill in missing atoms and residues and to perform protonation, and in-house routines for determining connectivity and formal charge states for proteins. For ligand components of the system, it uses RDKit.
:::

:::{admonition} Prepare Complex has more features!
:class: tip
This module also fills in missing atoms and residues where possible, annotates the TRC with full connectivity information, and has additional configurable behavior. See the {doc}`Prepare Complex module documentation <../rush.prepare_complex>` for more info.
:::

To perform an accurate calculation without incurring too high an execution cost, we can use helper functions to find the set of fragments within 5Å of the ligand:
```{code-block} python
# Find ligand fragment index + nearby pocket
lig_idx = trc.residues.seqs.index("MK1")
frag_idcs = trc.topology.get_fragments_near_fragment(lig_idx, 5.0) + [lig_idx]
```

Finally, we can submit the full interaction energy calculation, using the above information to limit the simulation to only that set of fragments:
```{code-block} python
from rush import exess

# EXESS only takes the Topology, not the full TRC
with open("1hsg_t.json", "w") as f:
    f.write(json.dumps(trc.topology.to_json(), indent=2))

out = exess.interaction_energy(
    "1hsg_t.json",
    lig_idx,
    frag_keywords=exess.FragKeywords(
        level="Dimer",
        included_fragments=frag_idcs,
    ),
    run_opts=RunOpts(name="Tutorial: Interaction Energy E2E - EXESS"),
    collect=True,
)

out_file, _ = exess.save_energy_outputs(out)
with open(out_file) as f:
    out_data = json.load(f)
print(f"Interaction energy: {out_data['qmmbe']['expanded_hf_energy']}")
```
>>>>>>> 01791af6000f243f0b54f3fbd35c775cd99b46f7
