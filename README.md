# Tengu-py: Python SDK for the QDX Tengu API

This package exposes a simple provider and CLI for the different tools exposed by the QDX Tengu GraphQL API.

## Usage

### As a library

```python
import json
from pathlib import Path

import tengu
from tengu import Arg

TOKEN = "your tengu access token"

# get our client to talk with the API
client = tengu.Provider(access_token=TOKEN)

# get newest module versions
modules = client.latest_modules()

# get desciption and usage module
print(modules[0]["description"])
print(modules[0]["usage"])

# get exact input arguments needed for module
print(modules[0]["ins"])

## running convert

# path to protein pdb with correct charges and protonation
protein_pdb = Path("./examples/4w9f_prepared_protein.pdb")

# get base64 encoded data

file_arg = client.upload_arg(protein_pdb)

convert_res = client.run("github:talo/tengu-prelude/77e44748f1d1e20c463ef34cc40178d4f656ef0a#convert", [
  Arg(value = "PDB"), file_arg
])

// res contains "id" - the instance id; and "outs" - the ids of the return values

// we can pass arguments by "id" reference or by value literal

pick_res = client.run("github:talo/tengu-prelude/f8e2e55d9bd428aa7f2bbe3f87c24775fa592b10#pick_conformer", [
  Arg( id =  res["outs"][0]["id"] ), Arg( value = 0 ) }
])

client.poll_module_instance(id)
// status, progress, logs, outs - out values will be null until module_instance is done

// retrieve object result once done
conformer = client.object(pick_res["outs"][0])
```

## Sample Hermes Run

This will perform a Hermes calculation on a (small < 1000 atoms) raw input file, without doing any transformations
A sample `params.json` can be found [here](https://talo.github.io/qdx-common/qdx_common/qc/struct.Params.html#example)

```python
instance = client.run(
    "github:talo/tengu-prelude/60662930969362ce73d321b38929dac878e4ec9f#hermes_raw",
    [
        tengu.Arg(value = json.load(open("params.json")))
    ],
    "GADI",
    {"walltime": 400, "gpus": 4},
)

# keep polling to see if module_instance is successful. This may take a while
completed_instance = client.poll_module_instance(instance["id"], n_retries=10, poll_rate=100)

# the result will be an object, so fetch from object store
client.object(completed_instance["outs"][0]["id"]) # will return the json energy results
```

## Sample Hermes Energy Run

This will perform a Hermes energy calculation on larger system, predefined to perform energy calculations.
You can create a "conformer.json" by using the convert module described above or by converting a pdb with [qdx-py](https://pypi.org/project/qdx-py/)

```python
default_model = {"method": "RIMP2", "basis": "cc-pVDZ", "aux_basis": "cc-pVDZ-RIFIT", "frag_enabled": True}

frag_keywords = {
    "dimer_cutoff": 25,
    "dimer_mp2_cutoff": 25,
    "fragmentation_level": 2,
    "method": "MBE",
    "monomer_cutoff": 30,
    "monomer_mp2_cutoff": 30,
    "ngpus_per_node": 1,
    "trimer_cutoff": 10,
    "trimer_mp2_cutoff": 10,
    "fragmented_energy_type": "TotalEnergy",
}

scf_keywords = {
    "convergence_metric": "diis",
    "dynamic_screening_threshold_exp": 10,
    "ndiis": 8,
    "niter": 40,
    "scf_conv": 0.000001,
}

instance = client.run(
    "github:talo/tengu-prelude/60662930969362ce73d321b38929dac878e4ec9f#hermes_energy",
    [
        tengu.Arg(value = json.load(open("conformer.json")))
        Arg(value = default_model),
        Arg(value = {"frag": frag_keywords, "scf": scf_keywords}),
    ],
    "GADI",
    {"walltime": 400, "gpus": 4},
)

# keep polling to see if module_instance is successful. This may take a while, > 10 mins
completed_instance = client.poll_module_instance(instance["id"], n_retries=10, poll_rate=100)

# the result will be an object, so fetch from object store
client.object(completed_instance["outs"][0]["id"]) # will return the json energy results
```

## Sample QP Run

```python
frag_keywords = {
    "dimer_cutoff": 25,
    "dimer_mp2_cutoff": 25,
    "fragmentation_level": 2,
    "method": "MBE",
    "monomer_cutoff": 30,
    "monomer_mp2_cutoff": 30,
    "ngpus_per_node": 1,
    "reference_fragment": 1,
    "trimer_cutoff": 10,
    "trimer_mp2_cutoff": 10,
    "fragmented_energy_type": "InteractivityEnergy",
}

scf_keywords = {
    "convergence_metric": "diis",
    "dynamic_screening_threshold_exp": 10,
    "ndiis": 8,
    "niter": 40,
    "scf_conv": 0.000001,
}

default_model = {"method": "RIMP2", "basis": "cc-pVDZ", "aux_basis": "cc-pVDZ-RIFIT", "frag_enabled": True}

qp_instances = client.qp_run(
    "github:talo/tengu-prelude/0986e4b23780d5e976e7938dc02a949185090fa1#qp_gen_inputs",
    "github:talo/tengu-prelude/0986e4b23780d5e976e7938dc02a949185090fa1#hermes_energy",
    "github:talo/tengu-prelude/0986e4b23780d5e976e7938dc02a949185090fa1#qp_collate",
    provider.upload_arg(Path("some.pdb")),
    provider.upload_arg(Path("some.gro")),
    provider.upload_arg(Path("some.sdf")),
    Arg(value = "sdf"),
    Arg(value = "MOL"), # id of ligand in pdb
    Arg(
        value = default_model,
    ),
    Arg(value = {"frag": frag_keywords, "scf": scf_keywords}),
    Arg(
        value = [
            ("GLY", 100), # map of amino acids of interest
        ],
    ),
    "GADI",
    {"walltime": 400},
    autopoll = (10, 100) # optionally configure polling to wait on the final instance,
                         # and clean up if any of the prior instances fails
)

# if you set autpoll, you will get the results of the qp_collate instance,
# otherwise you will get an array with all the spawned instances, and have to poll manually
completed_instance = client.poll_module_instance(qp_collate_instance[2]["id"])

# the result will be an object, so fetch from object store
client.object(completed_instance["outs"][0]["id"]) # will return the json qp results
```

## Local runner

We also provide a local executor, that will run modules locally, without making remote calls

First, you must have nix installed and configured with an access token for qdx projects.

Then you must install the tengu-runtime with `nix run github:talo/tengu#tengu-runtime -- install`

Finally, you can run locally with

```python
from tengu import LocalProvider

client = LocalProvider()

## you should be able to use client.run / client.object / client.module_instance / client.poll_module instance as normal
```
