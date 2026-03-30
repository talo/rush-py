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

### Using in Your Project

Add to your `pyproject.toml`:
```toml
[project]
dependencies = [
    "rush-py",
]
```

## Rush Setup

Use environment variables to configure access:
- `RUSH_TOKEN`: Put your token's value here
- `RUSH_PROJECT`: Put your project's UUID value here; can find it in the URL once selecting a project in the Rush UI
- `RUSH_ENDPOINT`: Use this to choose between staging and prod; if omitted, defaults to prod

You can also put `RUSH_TOKEN` and `RUSH_PROJECT` in a `.env` file instead of exporting them in every terminal session. rush-py looks for a `.env` file in the current working directory first, then falls back to `~/.rush/.env`. Environment variables always take priority over `.env` values.

```
# .env
RUSH_TOKEN=your-token-here
RUSH_PROJECT=your-project-id-here
```

## Quick Start

```python
from pathlib import Path

from rush import exess

topology_path = Path.cwd() / "thrombin_1c_t.json"

# For energy, the only mandatory argument is the Topology
run = exess.energy(topology_path)

# Fetch the results for direct access
result = run.fetch()

# Save the results
paths = run.save()
```

Outputs are saved under `<workspace_dir>/<PROJECT_ID>/` (default: current working directory). To customize the workspace location, call `rush.session.configure(workspace_dir=Path("..."))`.

```python
# For interaction_energy, second argument is reference fragment
result = exess.interaction_energy(topology_path, 1).fetch()

# Use export keywords to obtain additional information
result = exess.energy(
    topology_path,
    frag_keywords=None,  # MBE is not supported for CHELPG charges
    export_keywords=exess.ExportKeywords(export_chelpg_charges=True)
).fetch()

# QMMM requires Residues too
topology_path = "./6a5j_t.json"
residues_path = "./6a5j_r.json"
# Without calling `.fetch()` or `.save()` the run takes place asynchronously
# and a run object is returned
run = exess.qmmm(
    topology_path=topology_path,
    residues_path=residues_path,
    n_timesteps=500,
    qm_fragments=[0],
)
# The output is a QMMMResult object that contains the geometries for each timestep,
# which can be swapped into a Topology's geometry field
result = run.fetch()

# Get the full list of parameters and default arguments for a function
help(exess.energy)
help(exess.interaction_energy)
help(exess.optimization)
help(exess.qmmm)
```

See the [docs](https://talo.github.io/rush-py) for more information!

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

### Running Tests
Run the full suite with:
```bash
uv run pytest
```

Common focused invocations:
```bash
uv run pytest -m "not submits_rush_jobs"
uv run pytest -m submits_rush_jobs --force-run-slow
uv run pytest tests/test_exess_energy.py
uv run pytest tests/test_exess_energy.py --rush-workspace-dir /tmp/rush-py-workspaces
```

See the Terms of Service for use of the underlying Rush software at https://qdx.co/terms.
