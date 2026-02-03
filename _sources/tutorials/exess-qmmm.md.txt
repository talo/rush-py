# EXESS: QM/MM

This tutorial covers QM/MM/ML workflows (QMMM) with EXESS using **rush-py**.

:::{note}
You still need `RUSH_TOKEN` and `RUSH_PROJECT` in your environment, but as of February 1, 2026, they are only required on first use (not at import time).
:::

## Prerequisites

Set your Rush environment variables before running the client:

- `RUSH_TOKEN`
- `RUSH_PROJECT`
- `RUSH_ENDPOINT` (optional)

These examples use `tests/data/6a5j_t.json` and `tests/data/6a5j_r.json`.

## Common setup

```{code-block} python
:caption: setup.py

from pathlib import Path

from rush import exess
from rush.client import RunOpts, set_opts

set_opts(workspace_dir=Path.cwd() / "tutorial-runs")
DATA_DIR = Path.cwd() / "tests" / "data"
```

## Example: QMMM with topology + residues

This example follows `tests/test_exess_qmmm.py`. QMMM requires residues; pass topology and residues JSON files.

```{code-block} python
:caption: qmmm.py

from rush import exess
from rush.client import RunOpts, save_object

res = exess.qmmm(
    DATA_DIR / "6a5j_t.json",
    DATA_DIR / "6a5j_r.json",
    n_timesteps=500,
    qm_fragments=[6],
    ml_fragments=[],
    run_opts=RunOpts(
        name="Tutorial: EXESS QMMM",
        tags=["rush-py", "tutorial", "qmmm"],
    ),
    collect=True,
)

# Single-output results are returned directly
save_object(res["path"])
```

**Example output shape**:

```{code-block} python
{"path": "<uuid>", "size": 0, "format": "json"}
```

:::{tip}
You can also pass PDB/SDF/TRC paths directly; rush-py will convert them to topology/residue objects under the hood.
:::

## Minimal QMMM input (from EXESS docs)

The EXESS input docs show a minimal QMMM example using two water fragments. This is the same idea, expressed in **rush-py** and then written to JSON for submission.

```{code-block} python
:caption: minimal_qmmm.py

from pathlib import Path

from rush.exess import qmmm
from rush.mol import Element, Fragment, Residue, Residues, Topology

# Build a tiny system with two water fragments

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
    seq_ns=[1, 2],
    insertion_codes=["", ""],
)

Path("molecule_t.json").write_text(topology.to_json())
Path("molecule_r.json").write_text(residues.to_json())

qmmm(
    topology_path="molecule_t.json",
    residues_path="molecule_r.json",
    n_timesteps=100,
    qm_fragments=[0],
    mm_fragments=[1],
)
```

:::{note}
`exess.qmmm` defaults to `method="RestrictedHF"`, `basis="STO-3G"`, and `temperature_kelvin=290.0` unless overridden. See `docs/exess/input.md` for defaults.
:::
