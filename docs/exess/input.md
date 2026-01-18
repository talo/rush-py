# Input format

EXESS inputs are JSON files loosely based on MolSSI QCSchema. In EXESS, the molecular group is called `topology`, and input files use a `topologies` array to allow batched runs (multiple systems evaluated with the same driver/model/keywords).

## High-level structure

A typical EXESS input has the following top-level groups:

```json
{
  "topologies": [ { /* topology */ } ],
  "driver": "Energy",
  "model": { /* method and basis */ },
  "keywords": { /* method and runtime options */ },
  "system": { /* hardware options */ }
}
```

`system` and `keywords` are optional. Examples of complete inputs are in the examples page.

## topologies

`topologies` is an array of `topology` objects. Each `topology` includes the molecular data, fragmentation, and connectivity.

### topology fields

| Field | Brief |
| --- | --- |
| `geometry` | Flat array of XYZ coordinates (x0, y0, z0, x1, y1, z1, ...). |
| `symbols` | Array of atomic symbols matching the geometry list. |
| `fragment_formal_charges` | Formal charge per fragment; single fragment uses `[0]` by default. |
| `fragment_multiplicities` | Multiplicity per fragment; single fragment defaults to singlet. |
| `fragments` | Array of arrays containing zero-indexed atom indices for each fragment. |
| `connectivity` | Array of `[atom_i, atom_j, bond_order]` entries. Required for covalent fragmentation. |
| `xyz` | Path to an XYZ file used in place of `geometry` and `symbols`. |

#### geometry

Example layout:

```json
"geometry": [
  -4.3997, 1.0764, 7.7009,
  -3.9597, 0.2664, 7.3109,
  -4.4297, 0.9964, 8.7009
]
```

#### symbols

```json
"symbols": ["O", "H", "H"]
```

#### fragments

Fragment indices are zero-indexed. Example for five water fragments:

```json
"fragments": [
  [0, 1, 2],
  [3, 4, 5],
  [6, 7, 8],
  [9, 10, 11],
  [12, 13, 14]
]
```

#### connectivity

Connectivity is required when a covalent bond is broken across fragments. EXESS will add hydrogen caps to satisfy valences and remove them after the calculation.

```json
"connectivity": [
  [1, 4, 1],
  [9, 12, 1]
]
```

#### xyz

`xyz` can be used instead of `geometry` and `symbols`, including for fragmented systems:

```json
{
  "topologies": [
    { "xyz": "/path/to/water.xyz" }
  ]
}
```

## driver

The `driver` field selects the calculation type:

- `Energy`
- `Gradient`
- `Optimization`
- `Dynamics`

## model

The `model` group controls the level of theory and basis sets:

| Field | Type | Brief |
| --- | --- | --- |
| `method` | Method | Method used for the calculation. |
| `basis` | string | Primary basis set. |
| `aux_basis` | optional string | Auxiliary basis for RI methods. |
| `standard_orientation` | StandardOrientation | `FullSystem`, `None`, or `PerFragment`. |
| `force_cartesian_basis_sets` | bool | Force Cartesian basis functions for higher angular momenta. |

Documented method values include `RestrictedHF`, `UnrestrictedHF`, and `RestrictedRIMP2`. Additional methods appear in other EXESS docs and are tracked in the [mismatches page](mismatches).

## system

The `system` group controls hardware usage:

| Field | Type | Brief |
| --- | --- | --- |
| `max_gpu_memory_mb` | optional uint64 | Max GPU memory to allocate per GPU (MB). |
| `oversubscribe_gpus` | bool | Allow multiple processes per GPU (expert). |
| `teams_per_node` | uint32 | Number of worker teams per node. |
| `gpus_per_team` | optional uint32 | GPUs per worker team. |

Example:

```json
"system": {
  "max_gpu_memory_mb": 24000,
  "teams_per_node": 4,
  "gpus_per_team": 1
}
```

## keywords

`keywords` contains method- and run-specific settings. The available groups are:

- `scf`
- `frag`
- `guess`
- `optimization`
- `dynamics`
- `boundary`
- `ff`
- `log`
- `export`
- `ks_dft`
- `rtat`
- `debug`

See the keyword reference page for full details.

## Input conversion tools

The upstream docs mention several helpers for building EXESS inputs:

- `parley.py` (https://github.com/JorgeG94/parley_exess) converts between XYZ and EXESS JSON. It can also add minimal defaults for `Dynamics` and `Optimization` drivers.
- `tools/input_transformer/create_json_input.jl` in the EXESS repo is a Julia helper for generating RHF inputs:

```bash
julia -E 'include("create_json_input.jl"); create_input_rhf("input.xyz", "BASIS")'
```

Rush-py also supports working with TRC and Topology files; see the [Objects and TRC Files guide](../guides/03-objects-and-trc-files) for details.
