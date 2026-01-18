# Running EXESS

This page covers both the EXESS executable and the rush-py interface. The input format and keyword reference are documented in the input and keyword pages.

## CLI (EXESS executable)

### Environment and installation

EXESS needs the records directory and optional validation directory to be discoverable at runtime. The upstream installation guide describes these environment variables:

```bash
export EXESS_PATH="$YOUR_PATH/exess"
export EXESS_RECORDS_PATH=$EXESS_PATH/records
export EXESS_VALIDATION_PATH=$EXESS_PATH/validation
```

Additional runtime environment variables are listed in the reference page, including `EXESS_OUTPUT_PATH` and `EXESS_HDF5_OUTPUT_PATH` for output locations.

### Single-node runs

For simple single-node calculations, EXESS provides a `runexess` wrapper script that launches the calculation with MPI:

```bash
module load exess
runexess your_input_file.json -g NGPUS
```

If `-g NGPUS` is omitted, the script will use all available GPUs. Use `runexess --help` for details on the wrapper arguments.

### Multi-node runs (fragmentation)

Multi-node runs are used for fragmentation calculations. With Slurm, a typical launch looks like:

```bash
SCHEDULER=slurm
# NNODES
# NGPUS_PER_NODE

# NTASKS_PER_NODE = NGPUS_PER_NODE + 2
# NTASKS = NTASKS_PER_NODE * NNODES

module load exess
srun --nnodes=10 --ntasks=60 --ntasks-per-node=6 --gpus-per-node=4 exess input.json
```

With `mpirun`, you must compute the per-node and total task counts explicitly:

```bash
SCHEDULER=PBS

NNODES=3
NGPUS_PER_TEAM=4
NTEAMS_PER_NODE=1

# nprocs_per_node = 1 + (NGPUS_PER_TEAM + 1) * NTEAMS_PER_NODE
# total_nprocs = NNODES * nprocs_per_node

module load exess
mpirun -np ${total_nprocs} --bind-to core --map-by ppr:${nprocs_per_node}:node exess input.json
```

### Fragmentation team sizing

Fragmentation distributes work across teams, where one team is allocated to a single fragment. Team sizing is controlled by `system.teams_per_node` and `system.gpus_per_team` in the input file. For example, with eight GPUs per node and one GPU per fragment, set:

```bash
NTEAMS_PER_NODE=8
NGPUS_PER_TEAM=1
```

The number of MPI tasks per node is:

`nprocs_per_node = 1 + (NGPUS_PER_TEAM + 1) * NTEAMS_PER_NODE`

See the `system` section in the input docs for details.

## Rush Python client

### Authentication and setup

Set the Rush environment variables before running the client:

- `RUSH_TOKEN`
- `RUSH_PROJECT`
- `RUSH_ENDPOINT` (optional)

### Basic usage

The rush-py EXESS wrapper accepts the same topology input format (JSON), and exposes both a direct EXESS entry point and convenience wrappers:

```python
from rush import exess

# Direct wrapper
exess.exess("input_topology.json", collect=True)

# Convenience wrappers
exess.energy(...)
exess.interaction_energy(...)
exess.chelpg(...)
exess.optimization(...)
exess.qmmm(...)
```

By default, runs are asynchronous and return a run ID. Pass `collect=True` to wait for completion, or collect later:

```python
from rush.client import collect_run
from rush.exess import exess as run_exess

run_id = run_exess("input_topology.json")
result = collect_run(run_id)
```

### Run metadata and resources

Run metadata (name, tags, description, email notifications) is configured via `run_opts`. Resource hints can be provided via `run_spec`:

```python
from rush import exess
from rush.client import RunOpts, RunSpec

res = exess.energy(
    "input_topology.json",
    run_opts=RunOpts(name="example", tags=["exess"]),
    run_spec=RunSpec(storage=1000, gpus=1),
    collect=True,
)
```

### Outputs and object store paths

Rush returns outputs as object store references (UUID paths plus format info). Use the EXESS output helpers to download the results:

```python
files = exess.save_energy_outputs(res)
```

Details on output files and the JSON and HDF5 structures are in the [outputs page](outputs).
