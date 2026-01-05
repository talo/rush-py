# rush-py: Rush Python Client

## Installation

Install from PyPI:
```bash
pip install rush-py
```

If you manage dependencies with `uv`, add it to your project (updates `pyproject.toml` and `uv.lock`):
```bash
uv add rush-py
```

### Using in your project

Add to your `pyproject.toml`:
```toml
[project]
dependencies = [
    "rush-py",
]
```

## Rush setup

Use environment variables to configure access:
- `RUSH_TOKEN`: Put your token's value here
- `RUSH_PROJECT`: Put your project's UUID value here; can find it in the URL once selecting a project in the Rush UI
- `RUSH_ENDPOINT`: Use this to choose between staging and prod; if omitted, defaults to prod

## Quick start

```python
from pathlib import Path

from rush import exess
from rush.client import collect_run

topology_path = Path.cwd() / "thrombin_1c_t.json"

# For energy, the only mandatory argument is the Topology
result = exess.energy(topology_path, collect=True)
exess.save_energy_outputs(result)
```

Outputs are saved under `<workspace_dir>/<PROJECT_ID>/` (default: current working directory). To customize the workspace location, call `rush.client.set_opts(workspace_dir=Path("..."))`.

```python
# For interaction_energy, second argument is reference fragment
result = exess.interaction_energy(topology_path, 1, collect=True)

# For chelpg, charges are extracted from the HDF5 output and returned as a list
output_info, charges = exess.chelpg(topology_path, collect=True)

# QMMM requires Residues too
md_topology_path = "./6a5j_t.json"
md_residues_path = "./6a5j_r.json"
# Without `collect=True`, the run takes place asynchronously, and a run ID is returned
id = exess.qmmm(
    md_topology_path,
    md_residues_path,
    n_timesteps=500,
    qm_fragments=[0],
    free_atoms=[0],
)
# The output for qmmm is a geometries json; can swap into a Topology's geometry field
result = collect_run(id)

# Get the full list of parameters and default arguments for a function
help(exess.energy)
help(exess.interaction_energy)
help(exess.chelpg)
help(exess.qmmm)
```

See the [Getting Started guide](/docs/guides/01-getting-started.md) for more usage information.

## Development

You can develop this project using `pip` + `venv`, or `uv`.

### With `pip` + `venv`
```bash
git clone git@github.com:talo/rush-py.git
cd rush-py
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### With `uv`
```bash
git clone git@github.com:talo/rush-py.git
cd rush-py
uv sync
source .venv/bin/activate
```
