# Your First rush-py Run

## Background Info: Rush Modules

Rush consists of modules, which are individual programs or entry points for tools available on the platform. The rush-py client provides one primary function for each module that allows you to call it, and it supports a complete set of inputs and outputs for each module (in most cases). In some cases, this gives you more flexibility than the Rush web interface. The Rush Python client strives to stay as faithful as possible to the module's underlying capabilities and give you direct and transparent access to it.

## Import and Call a Module

Each Rush module has a Python submodule that provides support for it. For example, import files and functions from `rush.exess` to use functions and supporting infrastructure for the EXESS module:
```python
from rush import exess
```

You can then access the function that directly wraps EXESS as expected:
```python
from rush import exess
exess.exess("input_topology.json", collect=True)
```

Or the possibly more clear:
```python
from rush.exess import exess as run_exess
run_exess("input_topology.json", collect=True)
```

And you're done! The EXESS module has a rich set of capabilities and configuration, but by default will perform an energy calculation on the input topology molecule. To learn more about EXESS capabilities, see the {doc}`EXESS documentation<../exess/index>`. To learn about QDX Topology files, check the {doc}`Topologies section<../exess/topologies>`.

You can use [the topology file for a small protein from the 1KUW structure on the PDB](https://github.com/talo/rush-py/blob/main/tests/data/1kuw_t.json) as the input, or for very quick runs, our topology files for [benzene](https://github.com/talo/rush-py/blob/main/tests/data/benzene_t.json) or [ethane](https://github.com/talo/rush-py/blob/main/tests/data/ethane_t.json).

### Getting Results

Here's a complete example that runs an energy calculation and prints the result:

```python
import json
from rush import exess
from rush.client import save_object

res = exess.energy("benzene_t.json", collect=True)
output_file = save_object(res[0]["path"])

with open(output_file) as f:
    output_data = json.load(f)

print("EXPANDED HF ENERGY:")
print(output_data["qmmbe"]["expanded_hf_energy"])
```

### Asynchronous Runs and Collecting Runs

Rush modules can take a long time, so by default run asynchronously: the function that triggers the run will return once the run is submitted. In order to obtain the output synchronously for this same call, pass `collect=True` as we've done above. You can also collect the run later:
```python
from rush.client import collect_run
from rush.exess import exess as run_exess
id = run_exess("input_topology.json")
result = collect_run(id)
```

As shown, module calls return a "run ID" that can be used as above to collect any run that you've submitted.


### Supplementary Entrypoints

Or, call one of the other supplementary EXESS entry points, built to facilitate using EXESS's various capabilities:
```python
from rush import exess
exess.energy(...)
exess.interaction_energy(...)
exess.chelpg(...)
exess.optimization(...)
exess.qmmm(...)
```

## View your run in the Rush Web Interface

On the project page in the Rush web interface, your runs will appear on the left sidebar. To get more info, you can also go to https://rush.cloud/projects/{PROJECT_ID}/runs (after replacing the `{PROJECT_ID}` placeholder in the URL with your actual project ID) and click on any run to get debug-level information about the run.
