# Tutorial 8: Hyper Solvation

**What you get:** A solvated TRC structure ready for downstream minimization or dynamics.

| | |
|---|---|
| **Time** | ~2-5 minutes |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.12+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Why This Matters

Most molecular workflows need explicit solvent before minimization or MD. Solvation is easy to do manually once, but hard to do consistently across large batches. `hyper.hyper_solvate_sumo()` gives you a reproducible API-level step that can be scripted, tracked, and reused in larger pipelines.

---

## Quick Start

```python
from rush import TRC, hyper
from rush.client import RunOpts

run = hyper.hyper_solvate_sumo(
    ["valid_trc.json"],
    config=hyper.HyperConfig(
        max_inputs=8,
        padding_nm=0.8,
        seed=12345,
        timeout_seconds=120,
    ),
    run_opts=RunOpts(
        name="Tutorial: Hyper Solvate",
        tags=["rush-py", "tutorial", "hyper", "solvate"],
    ),
)

results = run.fetch()
item = results[0]

if isinstance(item, hyper.ItemError):
    raise RuntimeError(f"Hyper returned item error: {item}")

assert isinstance(item, TRC)
print(f"Solvated atoms: {len(item.topology.symbols)}")
```

This submits one structure, waits for completion with `fetch()`, and returns a parsed `TRC` object in memory.

---

## Reading the Output

`hyper_solvate_sumo()` returns `RushRun[hyper.TRCBatchResultRef]`.

- `run.collect()` gives a `TRCBatchResultRef`
- `result_ref.fetch()` returns `list[TRC | ItemError]`
- `result_ref.save()` returns `list[Path | ItemError]`

For production code, always branch on `ItemError` per item so partial batch failures are handled explicitly.

---

## Notes

- `padding_nm` controls solvent box padding around the solute.
- `seed` makes solvent placement deterministic across repeated runs.
- `max_inputs` should stay within module limits (1..=128).
- For batch workflows, submit multiple TRCs in one call and inspect each item independently.

---

## See Also

- {doc}`Hyper API Reference <../rush.hyper>`
- [Example script](https://github.com/talo/rush-py/tree/main/examples/hyper-solvate){target="_blank"}
- {doc}`NN-xTB tutorial <07-nnxtb-energy>` for a downstream energy workflow
