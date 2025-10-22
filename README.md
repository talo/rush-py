# rush-py2: Rush Python Bindings

## Project setup

You can use this project using `pip` + `venv`, or `uv`. Reach out if you need guidance.

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
uv venv
source .venv/bin/activate
pip install -e .

# Can also run directly (pass -h or --help for usage info); no need to enter the venv or run pip install!
uv run rush-exess-energy -h
uv run rush-exess-interaction-energy -h
uv run rush-exess-chelpg -h
uv run rush-exess-qmmm -h
```

## Rush setup

Use environment variables to configure access:
- `RUSH_ENDPOINT`: Use this to choose between staging and prod; if omitted, defaults to staging
- `RUSH_TOKEN`: Put your token's value here
- `RUSH_PROJECT`: Put your project's UUID value here; can find it in the URL once selecting a project in the Rush UI

## Usage

```python
from rush_py2 import exess

# Put the path to your Topology object here; can use pathlib.Path too if you like type safety
topology_path = "./thrombin_1c_t.json"

# for energy, the only mandatory argument is the Topology
output_path = exess.energy(topology_path)

# for interaction_energy, second argument is reference fragment
output_path = exess.interaction_energy(topology_path, 1)

# for chelpg, charges extracted from hdf5 and returned as a list
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
