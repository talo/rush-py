# Running NN-xTB

NN-xTB is accessed through the `rush.nnxtb` module in the Rush Python SDK.

## Prerequisites

- Python 3.12+
- `rush-py` installed (`uv pip install rush-py` or `pip install rush-py`)
- Rush credentials configured: `RUSH_TOKEN`, `RUSH_PROJECT` (and optionally `RUSH_ENDPOINT`) set as environment variables or in a `.env` file

## Basic usage

```python
from rush import nnxtb

# Submit and wait for results
result = nnxtb.energy("topology.json").fetch()
```

The input is a TRC (topology representation) JSON file containing atomic coordinates and element information. See the {doc}`EXESS topologies documentation <../exess/topologies>` for the full TRC format specification. You can also convert from PDB or SDF using the `rush.convert` module.

## Function signature

```python
def energy(
    topology_path: Path | str,
    compute_forces: bool | None = None,       # Default: True
    compute_frequencies: bool | None = None,   # Default: False
    multiplicity: int | None = None,           # Default: 1 (singlet)
    run_spec: RunSpec = RunSpec(gpus=1, storage=100),
    run_opts: RunOpts = RunOpts(),
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
### Return value

Returns a `Run[nnxtb.ResultRef]` handle. Call `.collect()` to wait for the lightweight result reference, or use `.fetch()` / `.save()` as shortcuts.

## Synchronous vs asynchronous

By default, `nnxtb.energy()` submits the job and returns immediately with a `Run` handle. This is useful for batch workflows where you want to submit many jobs and collect results later:

```python
from rush import nnxtb

# Submit asynchronously
run = nnxtb.energy("topology.json")

# ... do other work ...

# Collect when ready
result_ref = run.collect()
```

For interactive use, call `.fetch()` to block until the result is ready and parse it immediately:

```python
result = nnxtb.energy("topology.json").fetch()
```

## Run metadata

Use `RunOpts` to attach metadata to your runs. This makes them easier to find in the Rush web interface:

```python
from rush import RunOpts

run = nnxtb.energy(
    "topology.json",
    run_opts=RunOpts(
        name="Ligand screening batch 1",
        tags=["nnxtb", "screening", "project-x"],
    ),
)
```

## Parsing results

After submitting a run, either fetch the parsed `nnxtb.Result` in memory or save
the raw JSON output to the workspace:

```python
from rush import nnxtb

run = nnxtb.energy("topology.json")

# Parse into a structured object
result = run.fetch()
print(f"Energy: {result.energy_mev:.2f} meV")

if result.forces_mev_per_angstrom:
    print(f"Number of atoms: {len(result.forces_mev_per_angstrom)}")

if result.frequencies_inv_cm:
    print(f"Number of frequencies: {len(result.frequencies_inv_cm)}")
```

`nnxtb.Result` has three fields:
- `energy_mev`
- `forces_mev_per_angstrom`
- `frequencies_inv_cm`

To save the raw JSON output instead:

```python
from rush import nnxtb

paths = nnxtb.energy("topology.json").save()
```

## Error handling

If the submission fails (e.g., invalid input, authentication error), the function prints error messages to stderr. Common issues:

- **Missing credentials**: Ensure `RUSH_TOKEN` and `RUSH_PROJECT` are set
- **Invalid topology**: The input file must be a valid TRC JSON file
- **Timeout**: `Run.collect()` waits up to one hour by default before timing out
