# Your First rush-py Run

## Background Info: Rush Modules

Rush consists of modules, which are individual programs or entry points for tools available on the platform. The rush-py client provides one primary function for each module that allows you to call it, and it supports a complete set of inputs and outputs for each module (in most cases). In some cases, this gives you more flexibility than the Rush web interface. The Rush Python client strives to stay as faithful as possible to the module's underlying capabilities and give you direct and transparent access to it.

## Import and Call a Module

Each Rush module has a Python submodule that provides support for it. For example, import files and functions from `rush.exess` to use functions and supporting infrastructure for the EXESS module:
```python
from rush import exess
```

For example, single-point energy calculations can be performed using `exess.energy`:
```python
result = exess.energy("input_topology.json").fetch()
```

And you're done! The EXESS module has a rich set of capabilities and configuration, but by default will perform an energy calculation on the input topology molecule. To learn more about EXESS capabilities, see the [EXESS documentation](../exess/index.md). To learn about TRC files, check the [topologies section](../exess/topologies.md).

You can use [the topology file for a small protein from the 1KUW structure on the PDB](https://github.com/talo/rush-py/blob/main/tests/data/1kuw_t.json) as the input, or for very quick runs, our topology files for [benzene](https://github.com/talo/rush-py/blob/main/tests/data/benzene_t.json) or [ethane](https://github.com/talo/rush-py/blob/main/tests/data/ethane_t.json).

### Getting Results

Here's a complete example that runs an energy calculation and prints the result:

```python
import json
from rush import exess

result = exess.energy("benzene_t.json").fetch()
print("EXPANDED HF ENERGY:")
print(result.calc.qmmbe.expanded_hf_energy)
```

### Asynchronous Runs

Rush modules can take a long time, so by default they return a `Run` handle as soon as the run is submitted. To wait for completion and get the result, call `.fetch()`:
```python
from rush import exess

run = exess.energy("input_topology.json")  # run started
result = run.fetch()  # waits for completion
```

As shown, module calls return a `Run` object that can be used as above to fetch any run that you've submitted.


### Supplementary Entrypoints

Or, call one of the other supplementary EXESS entry points, built to facilitate using EXESS's various capabilities:
```python
from rush import exess

exess.energy(...)
exess.interaction_energy(...)
exess.optimization(...)
exess.qmmm(...)
```

## View your run in the Rush Web Interface

On the project page in the Rush web interface, your runs will appear on the left sidebar. To get more info, you can also go to https://rush.cloud/projects/{PROJECT_ID}/runs (after replacing the `{PROJECT_ID}` placeholder in the URL with your actual project ID) and click on any run to get debug-level information about the run.

## What's Next?

Now that you've run your first calculation, here are three paths forward:

### 🧪 Get Started with Tutorials

Learn by doing with step-by-step examples that walk you through real-world workflows:

- {doc}`CHELPG Charge Analysis <../tutorials/01-exess-chelpg>` — Understand molecular charge distributions
- {doc}`Single Point Energy <../tutorials/02-exess-spe>` — The foundational QM energy calculation
- {doc}`Exporting Results <../tutorials/03-exess-exports>` — Get data out of Rush for downstream use
- {doc}`Geometry Optimization <../tutorials/04-exess-optimization>` — Find minimum-energy structures
- {doc}`Interaction Energy <../tutorials/05-exess-interaction-energy>` — Quantify binding between molecules
- {doc}`QM/MM Calculations <../tutorials/06-exess-qmmm>` — Mixed quantum/classical simulations

### 📖 Understand Rush-Py Core

Deep dive into data structures and workflow mechanics:

- {doc}`Objects and TRC Files <03-objects-and-trc-files>` — How Rush represents molecular data
- {doc}`General Info <04-general-info>` — Platform concepts and conventions
- {doc}`../modules` — Complete API reference for all Rush modules

### 🔬 Advanced EXESS

Comprehensive guide to the EXESS electronic structure system:

- {doc}`EXESS Documentation <../exess/index>` — Full reference for inputs, keywords, methods, and outputs
