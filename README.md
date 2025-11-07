# rush-py2: Rush Python Bindings

## Project setup

You can use this project using `pip` + `venv`, or `uv`. Reach out if you'd like support for a different workflow!

### With `pip` + `venv`
```bash
git clone git@github.com:talo/rush-py2.git
cd rush-py2
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### With `uv`
```bash
git clone git@github.com:talo/rush-py2.git
cd rush-py2
uv sync
source .venv/bin/activate

# Run directly (pass -h/--help for usage info); no need to enter the venv
uv run rush-exess-energy [...]
uv run rush-exess-interaction-energy [...]
uv run rush-exess-chelpg [...]
uv run rush-exess-qmmm [...]
```

### With `pixi`
Add to `pixi.toml`
```toml
[pypi-dependencies]
rush-py2 = { git = "https://github.com/talo/rush-py2.git", rev = "main" }
```

### Using in your project

Add to your `pyproject.toml`:
```toml
[project]
dependencies = [
    "rush-py2 @ git+ssh://git@github.com/talo/rush-py2.git",
]
```

## Rush setup

Use environment variables to configure access:
- `RUSH_TOKEN`: Put your token's value here
- `RUSH_PROJECT`: Put your project's UUID value here; can find it in the URL once selecting a project in the Rush UI
- `RUSH_ENDPOINT`: Use this to choose between staging and prod; if omitted, defaults to staging

## Usage

```python
from rush_py2 import exess

# Can use pathlib.Path too if you like type safety
topology_path = "./thrombin_1c_t.json"

# For energy, the only mandatory argument is the Topology
output_path = exess.energy(topology_path)

# For interaction_energy, second argument is reference fragment
output_path = exess.interaction_energy(topology_path, 1)

# For chelpg, charges extracted from hdf5 and additionally returned as a list
output_path, charges = exess.chelpg(topology_path)

# QMMM requires Residues too
# the output for qmmm is a geometries json; can swap into a Topology's geometry field
md_topology_path = "./6a5j_t.json"
md_residues_path = "./6a5j_r.json"
output_path = exess.qmmm(
    md_topology_path,
    md_residues_path,
    n_timesteps=500,
    qm_fragments=[0],
    free_atoms=[0],
)

# get the full list of parameters and default arguments for a function
help(exess.energy)
help(exess.interaction_energy)
help(exess.chelpg)
help(exess.qmmm)
```

The outputs will be downloaded to the current folder in the form `{uuid}.json`.
