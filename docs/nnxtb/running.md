# Running NN-xTB

NN-xTB is accessed through the `rush.nnxtb` module in the Rush Python SDK.

## Prerequisites

- Python 3.12+
- `rush-py` installed (`uv pip install rush-py` or `pip install rush-py`)
- Rush credentials configured: `RUSH_TOKEN`, `RUSH_PROJECT` (and optionally `RUSH_ENDPOINT`) set as environment variables or in a `.env` file

## Basic usage

```python
from rush.nnxtb import nnxtb

# Submit and wait for results
outputs = nnxtb("topology.json", collect=True)
```

The input is a TRC (topology representation) JSON file containing atomic coordinates and element information. See the {doc}`EXESS topologies documentation <../exess/topologies>` for the full TRC format specification. You can also convert from PDB or SDF using the `rush.convert` module.

## Function signature

```python
def nnxtb(
    topology_path: Path | str,
    compute_forces: bool | None = None,       # Default: True
    compute_frequencies: bool | None = None,   # Default: False
    multiplicity: int | None = None,           # Default: 1 (singlet)
    run_spec: RunSpec = RunSpec(gpus=1, storage=100),
    run_opts: RunOpts = RunOpts(),
    collect: bool = False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `topology_path` | `Path \| str` | *required* | Path to a TRC topology JSON file |
| `compute_forces` | `bool \| None` | `True` | Compute per-atom forces |
| `compute_frequencies` | `bool \| None` | `False` | Compute vibrational frequencies (more expensive) |
| `multiplicity` | `int \| None` | `1` | Spin multiplicity (1 = singlet, 2 = doublet, ...) |
| `run_spec` | `RunSpec` | 1 GPU, 100 MB storage | Compute resources to request |
| `run_opts` | `RunOpts` | empty | Run metadata (name, tags, description, email) |
| `collect` | `bool` | `False` | If `True`, block until the run completes and return outputs |

### Return value

- **`collect=False`** (default): Returns a run ID. Use `collect_run(id)` later to retrieve results.
- **`collect=True`**: Blocks until the run completes and returns the output object (a dict with a `path` key pointing to the JSON result in the object store).

## Synchronous vs asynchronous

By default, `nnxtb()` submits the job and returns immediately with a run ID. This is useful for batch workflows where you want to submit many jobs and collect results later:

```python
from rush.nnxtb import nnxtb
from rush.client import collect_run

# Submit asynchronously
id = nnxtb("topology.json")

# ... do other work ...

# Collect when ready
outputs = collect_run(id)
```

For interactive use, pass `collect=True` to block until the result is ready:

```python
outputs = nnxtb("topology.json", collect=True)
```

## Run metadata

Use `RunOpts` to attach metadata to your runs. This makes them easier to find in the Rush web interface:

```python
from rush.client import RunOpts

outputs = nnxtb(
    "topology.json",
    run_opts=RunOpts(
        name="Ligand screening batch 1",
        tags=["nnxtb", "screening", "project-x"],
    ),
    collect=True,
)
```

## Parsing results

After collecting a run, either fetch the parsed `NnxtbResult` in memory or save
the raw JSON output to the workspace:

```python
from rush.nnxtb import NnxtbResult, fetch_outputs, nnxtb

outputs = nnxtb("topology.json", collect=True)

# Parse into a structured object
res: NnxtbResult = fetch_outputs(outputs)
print(f"Energy: {res.energy_mev:.2f} meV")

if res.forces_mev_per_angstrom:
    print(f"Number of atoms: {len(res.forces_mev_per_angstrom)}")

if res.frequencies_inv_cm:
    print(f"Number of frequencies: {len(res.frequencies_inv_cm)}")
```

`NnxtbResult` has three fields:
- `energy_mev`
- `forces_mev_per_angstrom`
- `frequencies_inv_cm`

To save the raw JSON output instead:

```python
from rush.nnxtb import nnxtb, save_outputs

outputs = nnxtb("topology.json", collect=True)
paths = save_outputs(outputs)
```

## Error handling

If the submission fails (e.g., invalid input, authentication error), the function prints error messages to stderr. Common issues:

- **Missing credentials**: Ensure `RUSH_TOKEN` and `RUSH_PROJECT` are set
- **Invalid topology**: The input file must be a valid TRC JSON file
- **Timeout**: `collect_run` waits up to one hour by default before timing out
