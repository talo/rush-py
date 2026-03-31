# Tutorial 8: Hyper Solvate Sumo

**What you get:** A solvated `TRC` structure that you can pass directly into minimization or MD.

| | |
|---|---|
| **Time** | ~2-5 minutes |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.12+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Quick Start

```python
from pathlib import Path

from rush import RunOpts, TRC, hyper

input_trc = Path("tests/data/hyper/valid_trc.json")

run = hyper.hyper_solvate_sumo(
    [input_trc],
    config=hyper.HyperConfig(max_inputs=8, padding_nm=0.8, seed=12345, timeout_seconds=120),
    run_opts=RunOpts(name="Tutorial: Hyper Solvate", tags=["rush-py", "tutorial", "hyper", "solvate"]),
)

item = run.fetch()[0]
if not isinstance(item, TRC):
    raise RuntimeError(f"Unexpected per-item error: {item}")

print("Solvated atoms:", len(item.topology.symbols))
```

---

## Output Contract

`hyper_solvate_sumo()` returns `Run[hyper.TRCBatchResultRef]`.

- `run.collect()` -> `TRCBatchResultRef`
- `result_ref.fetch()` -> `list[TRC | ItemError]`
- `result_ref.save()` -> `list[Path | ItemError]`

---

## See Also

- {doc}`Hyper Minimize tutorial <09-hyper-minimize_sumo>`
- {doc}`Hyper Run tutorial <10-hyper-run_sumo>`
- {doc}`Hyper API reference <../rush.hyper>`
