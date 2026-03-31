# Tutorial 10: Hyper Run Sumo

**What you get:** Trajectory and optional checkpoint artifacts from short molecular dynamics runs.

| | |
|---|---|
| **Time** | ~3-10 minutes |
| **Skill level** | Intermediate |
| **Prerequisites** | Python 3.12+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Quick Start

```python
from pathlib import Path

from rush import RunOpts, hyper

data_dir = Path("tests/data/hyper")

run = hyper.hyper_run_sumo(
    [
        hyper.RunInput(
            sim_config=data_dir / "sim_config.json",
            topology=data_dir / "methanol_topology.json",
            coordinates=data_dir / "methanol_trc.json",
        )
    ],
    config=hyper.HyperRunConfig(
        max_inputs=4,
        nsteps=20,
        dt_ps=0.001,
        temperature_k=300.0,
        ensemble="Nvt",
        minimize_before_run=False,
        solvate_before_run=False,
        use_gpu=False,
        nthreads=1,
        timeout_seconds=900,
    ),
    run_opts=RunOpts(name="Tutorial: Hyper Run", tags=["rush-py", "tutorial", "hyper", "run"]),
)

item = run.fetch()[0]
if isinstance(item, hyper.ItemError):
    raise RuntimeError(f"Hyper run failed: {item}")

print("Trajectory bytes:", len(item.trajectory))
print("Checkpoint bytes:", 0 if item.checkpoint is None else len(item.checkpoint))
```

---

## Output Contract

`hyper_run_sumo()` returns `Run[hyper.RunResultRef]`.

- `run.collect()` -> `RunResultRef`
- `result_ref.fetch()` -> `list[RunOutput | ItemError]`
- `result_ref.save()` -> `list[RunOutputPaths | ItemError]`

---

## See Also

- {doc}`Hyper Solvate tutorial <08-hyper-solvate_sumo>`
- {doc}`Hyper Minimize tutorial <09-hyper-minimize_sumo>`
- {doc}`Hyper Run API reference <../rush.hyper_run_sumo>`
