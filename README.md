# Tengu-py: Python SDK for the QDX Tengu API

This package exposes a simple provider and CLI for the different tools exposed by the QDX Tengu GraphQL API.

## Usage

### As a library

``` python
import json
from pathlib import Path

import tengu

TOKEN = "your qdx access token"

# get our client to talk with the API
client = tengu.Provider(access_token=TOKEN)

# get modules that can be run
client.modules()

## running convert

# path to protein pdb with correct charges and protonation
protein_pdb = Path("./examples/4w9f_prepared_protein.pdb")

# get base64 encoded data

file_arg = provider.upload_arg(protein_pdb)

res = client.run("github:talo/tengu-prelude/f8e2e55d9bd428aa7f2bbe3f87c24775fa592b10#convert", [ 
{ "value": "PDB" }, file_arg
])

// res contains "id" - the instance id; and "outs" - the ids of the return values 

// we can pass arguments by "id" reference or by value literal

client.run("github:talo/tengu-prelude/f8e2e55d9bd428aa7f2bbe3f87c24775fa592b10#pick_conformer", [ 
{ "id": res["outs"][0]["id"] }, { "value": 1 }
])

client.poll_module_instance(id) 
// status, progress, logs, outs - out values will be null until module_instance is done
```

## Sample QP Run

``` python
provider.qp_run(
    "github:talo/tengu-prelude/0986e4b23780d5e976e7938dc02a949185090fa1#qp_gen_inputs",
    "github:talo/tengu-prelude/0986e4b23780d5e976e7938dc02a949185090fa1#hermes_energy",
    "github:talo/tengu-prelude/0986e4b23780d5e976e7938dc02a949185090fa1#qp_collate",
    provider.upload_arg(Path("some.pdb")),
    provider.upload_arg(Path("some.gro")),
    provider.upload_arg(Path("some.sdf")),
    Arg(None, "sdf"),
    Arg(None, "MOL"),
    Arg(
        None,
        default_model,
    ),
    Arg(None, {"frag": frag_keywords, "scf": scf_keywords}),
    Arg(
        None,
        [
            ("GLY", 993),
            ("ASP", 994),
            ("VAL", 863),
            ("LYS", 882),
            ("TYR", 931),
            ("GLY", 935),
            ("VAL", 911),
            ("GLU", 930),
            ("ALA", 880),
            ("LEU", 983),
            ("PRO", 933),
            ("LEU", 855),
            ("MET", 929),
            ("SER", 936),
            ("LEU", 932),
        ],
    ),
    "GADI",  # "NIX",
    {"walltime": 420},
)
```
