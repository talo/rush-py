# EXESS Geometry Optimization

This tutorial covers geometry optimization with EXESS using **rush-py**, based on the EXESS docs and the test suite.

:::{note}
You still need `RUSH_TOKEN` and `RUSH_PROJECT` in your environment, but as of February 1, 2026, they are only required on first use (not at import time).
:::

## Prerequisites

Set your Rush environment variables before running the client:

- `RUSH_TOKEN`
- `RUSH_PROJECT`
- `RUSH_ENDPOINT` (optional)

These examples use `tests/data/benzene_t.json`.

## Common setup

```{code-block} python
:caption: setup.py

from pathlib import Path

from rush import exess
from rush.client import RunOpts, set_opts

set_opts(workspace_dir=Path.cwd() / "tutorial-runs")
DATA_DIR = Path.cwd() / "tests" / "data"
```

## Example: QM optimization (RI-MP2)

This example follows `tests/test_exess_optimization_qm.py` and matches the RI-MP2 optimization model in the EXESS docs. Optimization requires `max_iters`.

```{code-block} python
:caption: optimization_qm.py

from rush import exess
from rush.client import RunOpts, save_object

res = exess.optimization(
    max_iters=100,
    topology_path=DATA_DIR / "benzene_t.json",
    optimization_keywords=exess.OptimizationKeywords(),
    method="RestrictedRIMP2",
    basis="cc-pVDZ",
    aux_basis="cc-pVDZ-RIFIT",
    standard_orientation="None",
    run_opts=RunOpts(
        name="Tutorial: EXESS Optimization (QM)",
        tags=["rush-py", "tutorial", "optimization", "QM"],
    ),
    collect=True,
)

# Optimization outputs are returned as a list of object-store paths
for res_i in res:
    save_object(res_i["path"])
```

:::{tip}
Setting `standard_orientation="None"` prevents EXESS from rotating/translating the input geometry.
:::

## Example: ML/MM optimization (LBFGS)

The non-QM optimization example below follows `tests/test_exess_optimization.py`. The values shown for `optimization_keywords` are the only supported ones for non-QM runs.

```{code-block} python
:caption: optimization_ml.py

from rush import exess
from rush.client import RunOpts, save_object

res = exess.optimization(
    max_iters=100,
    topology_path=DATA_DIR / "benzene_t.json",
    optimization_keywords=exess.OptimizationKeywords(
        coordinate_system="Cartesian",
        algorithm="LBFGS",
        lbfgs_keywords=exess.LBFGSKeywords(),
    ),
    basis="STO-2G",
    standard_orientation="None",
    qm_fragments=[],
    mm_fragments=[],
    run_opts=RunOpts(
        name="Tutorial: EXESS Optimization (ML)",
        tags=["rush-py", "tutorial", "optimization", "ML"],
    ),
    collect=True,
)

for res_i in res:
    save_object(res_i["path"])
```

:::{note}
Optimization requires gradient-capable methods (RHF, RI-HF, or RI-MP2). See `docs/exess/electronic-structure-methods.md` for method support details.
:::
