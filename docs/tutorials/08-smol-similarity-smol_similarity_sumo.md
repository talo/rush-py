# Tutorial 8: smol-similarity nearest-neighbor search

**What you get:** For each query SMILES, a ranked list of nearest neighbors from one or more partition files plus tanimoto similarity scores.

| | |
|---|---|
| **Time** | ~1-2 minutes |
| **Skill level** | Beginner |
| **Prerequisites** | Python 3.12+, `rush-py` installed, `RUSH_TOKEN` and `RUSH_PROJECT` set |

---

## Quick Start

```python
from rush import RunOpts, smol_similarity

run = smol_similarity.smol_similarity_sumo(
    smol_partitions=["smi_partition1.json"],
    input_smis="input_smis.json",
    config=smol_similarity.SmolSimilarityConfig(
        min_similarity=0.0,
        min_results=1,
        max_results=10,
    ),
    run_opts=RunOpts(
        name="Tutorial: smol_similarity_sumo",
        tags=["rush-py", "tutorial", "smol-similarity"],
    ),
)

results = run.fetch()
```

`results` is a list with one entry per query SMILES. Each entry is either:
- `smol_similarity.Result` (successful query)
- `smol_similarity.ExecutionError` (query-level execution failure)

If top-level validation fails (for example empty query list or invalid config bounds), `run.collect()`/`run.fetch()` raises a Rush run error containing the validation message.

---

## Input files

Both inputs are JSON object files uploaded to Rush:

- `smol_partitions`: one or more JSON arrays of library SMILES strings
- `input_smis`: a JSON array of query SMILES strings

Example `input_smis.json`:

```json
[
  "c1cc(OC)ccc1C"
]
```

Example `smi_partition1.json`:

```json
[
  "Nc1cc(c[nH]c1=O)C(F)(F)Cl",
  "Nc1cc(c[nH]c1=O)C(F)(F)Br",
  "Nc1cc(c[nH]c1=O)C(F)(F)I"
]
```

---

## Reading results

```python
for i, item in enumerate(results):
    if isinstance(item, smol_similarity.ExecutionError):
        print(f"Query {i} failed at {item.stage}: {item.message}")
        continue

    print(f"Query {i}: {len(item.smiles)} matches")
    for rank, (smi, score) in enumerate(zip(item.smiles, item.similarities), start=1):
        print(f"  {rank:02d}. {smi}  score={score:.4f}")
```

The output is already ranked from most similar to least similar.

---

## Saving raw outputs

If you want the raw JSON objects in your workspace:

```python
saved = run.save()
for item in saved:
    if isinstance(item, smol_similarity.ExecutionError):
        continue
    print(item.output)
```

---

## See also

- API reference: {doc}`../rush.smol-similarity`
- Runnable example: `examples/smol-similarity-smol_similarity_sumo/`
