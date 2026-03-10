# Running EXESS

This page covers both the EXESS executable and the rush-py interface. The input format and keyword reference are documented in the input and keyword pages.

## Quick start (CLI vs rush-py)

```{eval-rst}
.. tab-set::

   .. tab-item:: EXESS CLI

      .. code-block:: bash

         runexess input.json -g 1

   .. tab-item:: rush-py

      .. code-block:: python

         from rush import exess

         exess.energy("input.json", collect=True)
```

## CLI (EXESS executable)

### Environment and installation

EXESS needs the records directory and optional validation directory to be discoverable at runtime. Set these environment variables:

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

### Input conversion (parley.py)

Upstream docs point to the `parley.py` tool for converting XYZ to EXESS JSON (and back). It can also attach minimal default keywords for Dynamics and Optimization. Basic usage:

```bash
usage: parley.py [-h] [--input_format {xyz,json}] [--output_format {json,xyz}] --input_file INPUT_FILE [--output_file OUTPUT_FILE]
                 [--basis_set BASIS_SET] [--aux_basis_set AUX_BASIS_SET] [--driver DRIVER] [--method METHOD]
```

Defaults:

- `input_format`: `xyz`
- `output_format`: `json`
- `basis_set`: `6-31G`
- `aux_basis_set`: `none`
- `driver`: `Energy` (options: Energy, Gradient, Dynamics, Optimization)
- `method`: `RestrictedHF` (options: RestrictedHF, RestrictedRIMP2)

The tool does not validate basis set choices; use the supported basis list in the [reference page](reference).

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

Sample topology inputs are available in `tests/data/`, including `tests/data/1kuw_t.json` (small protein topology), `tests/data/benzene_t.json`, and `tests/data/ethane_t.json`.

`exess.optimization` requires `max_iters` and does not support fragment-based QM calculations; fragments can still be used to define QM/MM/ML regions when needed.

To inspect function signatures and parameter docs locally, use Python's `help`:

```python
help(exess.energy)
help(exess.FragKeywords)
```

By default, runs are asynchronous and return a run ID. Pass `collect=True` to wait for completion, or collect later:

```python
from rush.client import collect_run
from rush.exess import exess as run_exess

run_id = run_exess("input_topology.json")
result = collect_run(run_id)
```

`collect_run` waits up to one hour by default before timing out.

### Run metadata and resources

Run metadata (name, tags, description, email notifications) is configured via `run_opts`. Resource hints can be provided via `run_spec`:

```python
from rush import exess
from rush.client import RunOpts, RunSpec

res = exess.energy(
    "input_topology.json",
    run_opts=RunOpts(name="example", tags=["exess"], email=True),
    run_spec=RunSpec(storage=1000, gpus=1),
    collect=True,
)
```

### Automatic file conversion

When a Rush module expects paths to Topology/Residues/Chains objects, rush-py can accept a PDB or SDF file path instead (proteins and ligands respectively), or a TRC file on disk. This is convenient for QMMM workflows.

### Uploading, downloading, and saving outputs

Rush uses object store paths for inputs and outputs. You can upload, download, and save objects explicitly:

```python
from rush.client import download_object, save_json, save_object, upload_object
```

The `save_outputs` helpers download outputs to the local workspace and preserve the original output signature, replacing object store paths with local paths. You do not need to download outputs when chaining module runs: object store paths can be passed directly as inputs.

Not every module has a `save_outputs` helper yet; if you rely on this pattern and find a gap, file an issue so it can be prioritized.

### Workspaces

`save_outputs` writes files into a per-project workspace directory, keeping a `history.json` ledger of module runs (run ID, time created, module revision). To customize the workspace location:

```python
from pathlib import Path
from rush import client

client.set_opts(workspace_dir=Path("/path/to/workspace"))
```

### View runs in the Rush web interface

Runs appear in the Rush web UI. For detailed debugging information, visit:

`https://rush.cloud/projects/{PROJECT_ID}/runs`

Replace `{PROJECT_ID}` with your actual project ID.

### Outputs and object store paths

Rush returns outputs as object store references (UUID paths plus format info). Use the EXESS output helpers to download the results:

```python
files = exess.save_energy_outputs(res)
```

Details on output files and the JSON and HDF5 structures are in the [outputs page](outputs).

### Support and feedback (rush-py)

If a rush-py module behaves unexpectedly or violates the documented client design, open an issue at:

`https://github.com/talo/rush-py/issues/new`

For EXESS feature requests, use the [EXESS feature request form](https://exess.qdx.co/requests?tab=feature).

For general feedback across Rush, a public feedback form is also available:

`https://forms.gle/1DPWK91utzJ6SED47`
