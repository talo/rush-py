# EXESS Interaction Energy

This tutorial walks through interaction energy calculations with EXESS using **rush-py**. The examples are adapted from the EXESS docs and the test suite, with an optional preparation flow based on the `demos/` folder.

:::{note}
You still need `RUSH_TOKEN` and `RUSH_PROJECT` in your environment, but as of February 1, 2026, they are only required on first use (not at import time).
:::

## Prerequisites

Set your Rush environment variables before running the client:

- `RUSH_TOKEN`
- `RUSH_PROJECT`
- `RUSH_ENDPOINT` (optional)

All examples below reference test fixtures in `tests/data/`.

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

This example follows `tests/test_exess_interaction_energy.py`. It computes interaction energy between a ligand fragment and the rest of the system, using nearby fragments for the interaction region.

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
    ),
    collect=True,
)

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
