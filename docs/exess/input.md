# Input format

EXESS inputs are JSON files loosely based on MolSSI QCSchema. In EXESS, the molecular group is called `topology`, and input files use a `topologies` array to allow batched runs (multiple systems evaluated with the same driver/model/keywords).

The schema below is compiled from the upstream EXESS docs, the EXESS/libqdx C++ schema (`libqdx.hpp`), the libqdx Rust types, and the rush-py interface.

## Schema overview

Top-level EXESS input fields (as parsed by EXESS/libqdx):

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | No | Defaults to `0.2.0`. |
| `topologies` | array of Topology | Yes | One or more molecular systems. |
| `residues` | array of Residues | No | TRC residue definitions, usually aligned to `topologies`. |
| `external_charges` | object | No | External point charges (positions + charges). |
| `model` | object | Yes | Method + basis configuration. |
| `system` | object | No | Hardware configuration. |
| `keywords` | object | Yes | Calculation parameters; may be `{}`. |
| `driver` | string | Yes | Calculation type (Energy, Gradient, Dynamics, Optimization, Hessian, QMMM). |
| `title` | string | No | Printed in output files. |
| `check_schema` | bool | No | Defaults to true when schema checks are enabled. |

`keywords` is required in the C++ schema, but can be an empty object because defaults are applied by the parser.

## Default resolution order

Defaults are applied in the following order:

1. rush-py defaults: any non-`None` values set in Python (function defaults or dataclass defaults) are explicit values.
2. EXESS/libqdx JSON parser defaults (as defined in `libqdx.hpp`) for omitted fields.
3. EXESS internal defaults for values that remain unset after parsing.

## topologies

`topologies` is an array of `topology` objects. Each `topology` contains the molecular data, fragmentation, and connectivity. Batched mode is achieved by listing multiple topologies.

### topology fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | No | Defaults to `0.2.0`. |
| `symbols` | array of string | Yes* | Atomic symbols ("H", "C", etc). Required if `xyz` is not provided. |
| `geometry` | array of float | Yes* | Flat XYZ array (length = 3 * atoms). Required if `xyz` is not provided. |
| `xyz` | string | Yes* | Path to XYZ file; EXESS will read it instead of `symbols` + `geometry`. |
| `velocities` | array of float | No | Flat XYZ velocity array (Angstrom/ps). |
| `labels` | array of string | No | Atom labels. |
| `partial_charges` | array of float | No | Per-atom partial charges. |
| `formal_charges` | array of int | No | Per-atom formal charges. Defaults to 0 for all atoms. |
| `connectivity` | array of bonds | No | Bond list for covalent fragmentation. |
| `stereochemistry` | array of int | No | One entry per bond. |
| `fragments` | array of arrays | No | Zero-indexed atom indices per fragment. If omitted, EXESS assumes a single fragment containing all atoms. |
| `fragment_formal_charges` | array of int | No | Per-fragment formal charges. Defaults to 0 per fragment if omitted. |
| `fragment_partial_charges` | array of float | No | Per-fragment partial charges. |
| `fragment_multiplicities` | array of int | No | Per-fragment multiplicities. Defaults to 1 per fragment if omitted. |
| `waters` | array of two ints | No | First/last water molecule treated classically. |


The EXESS C++ schema requires either `symbols` + `geometry` or `xyz`, but not both.

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

#### xyz

`xyz` can be used instead of `geometry` and `symbols`, including for fragmented systems:

```json
{
  "topologies": [
    { "xyz": "/path/to/water.xyz" }
  ]
}
```

#### connectivity

Connectivity is required when a covalent bond is broken across fragments. EXESS will add hydrogen caps to satisfy valences and remove them after the calculation.

A bond entry is `[atom_index_1, atom_index_2, bond_order]`.

EXESS/libqdx defines the following bond order values:

- `1`: Single
- `2`: Double
- `3`: Triple
- `4`: Quadruple
- `5`: Quintuple
- `6`: Sextuple
- `250`: FiveAndAHalf
- `251`: FourAndAHalf
- `252`: ThreeAndAHalf
- `253`: TwoAndAHalf
- `254`: OneAndAHalf
- `255`: Ring

Example:

```json
"connectivity": [
  [1, 4, 1],
  [9, 12, 1]
]
```

#### stereochemistry

EXESS/libqdx accepts a list of integers aligned with the `connectivity` list:

- `0`: None
- `1`: Up
- `-1`: Down
- `2`: Either
- `3`: CisOrTrans

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

#### fragment_formal_charges / fragment_multiplicities

- If `fragment_formal_charges` is omitted, EXESS sets fragment charges to the sum of per-atom `formal_charges` when provided, otherwise 0.
- If `fragment_multiplicities` is omitted, EXESS sets multiplicity to 1 for every fragment.
- If `fragments` is omitted, EXESS assumes a single fragment containing all atoms and derives defaults accordingly.

#### waters

`waters` is a two-element array identifying the first and last water molecule to be treated classically (used by some QMMM/Q4ML workflows).

## residues

`residues` is an optional array of residue definitions used for TRC-based workflows and QMMM. Each element is a `Residues` object:

| Field | Type | Notes |
| --- | --- | --- |
| `labeled` | array[int] | Optional list of residue indices with labels. |
| `labels` | array of array[string] | Optional labels per residue. |
| `residues` | array of array[int] | Atom indices per residue. |
| `seqs` | array[string] | Residue names. |
| `seq_ns` | array[int] | Residue numbers. |
| `insertion_codes` | array[string] | PDB-style insertion codes. |

## external_charges

External charges are represented as:

```json
"external_charges": {
  "positions": [0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
  "charges": [0.5, -0.5]
}
```

The positions array is a flat XYZ list, with one charge per position.

## driver

The `driver` field selects the calculation type:

- `Energy`
- `Gradient`
- `Dynamics`
- `Optimization`
- `Hessian`
- `QMMM`

## model

The `model` group controls the level of theory and basis sets:

| Field | Type | Notes |
| --- | --- | --- |
| `method` | string | Method used for the calculation (see below). |
| `basis` | string | Primary basis set. |
| `aux_basis` | string | Auxiliary basis for RI methods. |
| `standard_orientation` | string | `FullSystem`, `None`, or `PerFragment`. Default: `FullSystem`. |
| `force_cartesian_basis_sets` | bool | Default: true. |

Methods in the EXESS/libqdx schema:

- `RestrictedHF`
- `UnrestrictedHF`
- `RestrictedKSDFT`
- `RestrictedRIMP2`
- `UnrestrictedRIMP2`
- `RestrictedRICCSD`

The libqdx Rust parser accepts common aliases like `RHF`, `UHF`, `KSDFT`, `RIMP2`, and `RICCSD`.

## system

The `system` group controls hardware usage:

| Field | Type | Notes |
| --- | --- | --- |
| `max_gpu_memory_mb` | uint64 | Max GPU memory per process in MB. |
| `oversubscribe_gpus` | bool | Allow multiple processes per GPU. Default: false. |
| `teams_per_node` | uint32 | Worker teams per node. Default: 1. |
| `gpus_per_team` | uint32 | GPUs per team (overridable by `MBE_NGPUS`). |

## keywords

`keywords` contains method- and run-specific settings. See the keyword reference page for full details. The C++ schema expects `keywords` to be present even if empty.

## Rush-py input mapping

The Rush Python client does not submit the full EXESS input JSON directly. Instead, it accepts a topology path (and sometimes residues path) plus keyword objects and constructs the EXESS params internally.

Key differences:

- `Model` in rush-py only includes `standard_orientation` and `force_cartesian_basis_sets`; `method`, `basis`, and `aux_basis` are function parameters.
- `keywords` in rush-py are passed as Python dataclasses (for `SCFKeywords`, `FragKeywords`, `ExportKeywords`, `OptimizationKeywords`, etc.).
- `frag_keywords` defaults to a dimer fragmentation setup; pass `frag_keywords=None` to run a full-system calculation.
- `external_charges` and some keyword groups (e.g., `rtat`, `integrals`, `ks_dft`) are not yet exposed in the rush-py API.

See the Rush guides for TRC objects and conversions: [Objects and TRC Files](../guides/03-objects-and-trc-files).

## Rush-py defaults

Default values set by the rush-py entry points:

- `exess.exess` / `exess.energy` / `exess.interaction_energy`: `driver="Energy"` (for `exess.exess`), `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`, `standard_orientation` unset (EXESS default `FullSystem`), `force_cartesian_basis_sets` unset (EXESS default `true`).
- `exess.chelpg`: `method="RestrictedHF"`, `basis="cc-pVDZ"`, `standard_orientation="None"`, `force_cartesian_basis_sets=false`.
- `exess.qmmm`: `method="RestrictedHF"`, `basis="STO-3G"`, `aux_basis=None`, `standard_orientation` unset (EXESS default `FullSystem`), `force_cartesian_basis_sets` unset (EXESS default `true`), `dt_ps=0.002`, `temperature_kelvin=290.0`, `pressure_atm=None`, gradient method `Analytical` with default step size.
- `exess.optimization`: `method="RestrictedHF"`, `basis="cc-pVDZ"`, `aux_basis=None`, `standard_orientation` unset (EXESS default `FullSystem`), `force_cartesian_basis_sets` unset (EXESS default `true`), `max_iters` required.

Keyword defaults for rush-py are documented in the keyword reference page.

## Input conversion tools

The upstream docs mention several helpers for building EXESS inputs:

- `parley.py` (https://github.com/JorgeG94/parley_exess) converts between XYZ and EXESS JSON. It can also add minimal defaults for `Dynamics` and `Optimization` drivers.
- `tools/input_transformer/create_json_input.jl` in the EXESS repo is a Julia helper for generating RHF inputs:

```bash
julia -E 'include("create_json_input.jl"); create_input_rhf("input.xyz", "BASIS")'
```
