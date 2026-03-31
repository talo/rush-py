# Tutorial 9: Hyper Minimize Sumo

**What you get:** An energy-minimized structure from a structure/topology pair.

| | |
|---|---|
| **Time** | ~2-8 minutes |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.12+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Quick Start

```python
from pathlib import Path

from rush import RunOpts, TRC, hyper

data_dir = Path("tests/data/hyper")

run = hyper.hyper_minimize_sumo(
    [
        hyper.MinimizeInput(
            structure=data_dir / "methanol_trc.json",
            topology=data_dir / "methanol_topology.json",
        )
    ],
    config=hyper.HyperMinimizeConfig(max_inputs=4, steps=100, gtol=100.0, timeout_seconds=900),
    run_opts=RunOpts(name="Tutorial: Hyper Minimize", tags=["rush-py", "tutorial", "hyper", "minimize"]),
)

item = run.fetch()[0]
if not isinstance(item, TRC):
    raise RuntimeError(f"Unexpected per-item error: {item}")

print("Minimized atoms:", len(item.topology.symbols))
```

---

## Output Contract

`hyper_minimize_sumo()` returns `Run[hyper.TRCBatchResultRef]`.

- `run.collect()` -> `TRCBatchResultRef`
- `result_ref.fetch()` -> `list[TRC | ItemError]`
- `result_ref.save()` -> `list[Path | ItemError]`

---

## See Also

- {doc}`Hyper Solvate tutorial <08-hyper-solvate_sumo>`
- {doc}`Hyper Run tutorial <10-hyper-run_sumo>`
- {doc}`Hyper Minimize API reference <../rush.hyper_minimize_sumo>`
