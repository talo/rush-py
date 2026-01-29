# Topologies and residues

This page contains the full reference for `topologies` and `residues`. The
[input format page](input) provides a high-level summary with minimal examples.

(topologies)=
## topologies

`topologies` is an array of `topology` objects. Each `topology` contains the molecular data, fragmentation, and connectivity for a single system. Batched mode is achieved by listing multiple topologies with a shared driver/model/keywords configuration.

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
| `fragment_multiplicities` | array of int | No | Per-fragment multiplicities (open-shell systems). Defaults to 1 per fragment if omitted. |
| `waters` | array of two ints | No | First/last water molecule treated classically. |


Inputs require either `symbols` + `geometry` or `xyz`, but not both.

Notes:

- `labels` can be used to annotate specific atoms (e.g., alpha carbons in residues).
- `partial_charges`, `formal_charges`, and `labels` must match the length of `symbols` if provided. `partial_charges` are in atomic units.
- `connectivity` is a perceived bond list and should not be treated as a definitive source of bond information.
- `stereochemistry` is only valid if `connectivity` is present and must match its length.
- `velocities` are used for dynamics; they are present only in v1 inputs but are supported.
- `fragments` must include each atom exactly once when provided.
- rush-py expects topology JSON inputs (symbols + geometry). It does not accept `xyz` paths.

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

`xyz` can be used instead of `geometry` and `symbols`, including for fragmented systems (it is compatible with fragmentation keywords). Standard XYZ files begin with the atom count, followed by a comment line and then element + XYZ positions:

```text
3

O -1.1570 0.0630 -0.4817
H -1.1570 0.8180 -1.0766
H -1.1570 -0.6920 -1.0766
```

Example usage in EXESS:

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

EXESS defines the following bond order values (single/double/triple bonds are the most common values):

- `0`: Unspecified
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

A covalent fragmentation example with two ethane fragments connected across a C-C bond uses `connectivity: [[1, 4, 1]]` to signal a single bond between atom indices 1 and 4. EXESS uses this information to apply hydrogen caps on broken valences during fragmentation and removes them after the calculation.

#### stereochemistry

EXESS accepts a list of integers aligned with the `connectivity` list:

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

Example for a single, unfragmented system:

```json
{
  "topologies": [
    {
      "geometry": [
        -4.3997, 1.0764, 7.7009,
        -3.9597, 0.2664, 7.3109,
        -4.4297, 0.9964, 8.7009
      ],
      "symbols": ["O", "H", "H"],
      "fragment_formal_charges": [0],
      "fragment_multiplicities": [1]
    }
  ]
}
```

#### topology validation checks

Validation checks when constructing inputs:

- `geometry` must be length `3 * len(symbols)`.
- `labels`, `partial_charges`, and `formal_charges` must match `symbols` length if present.
- `connectivity` atom indices must be valid and may not reference the same atom twice.
- `stereochemistry` is only valid when `connectivity` is present and must be the same length.
- `velocities` must be length `3 * len(symbols)` if present.
- `fragments` must not duplicate atoms; each atom should appear in exactly one fragment.
- `fragment_formal_charges`, `fragment_partial_charges`, and `fragment_multiplicities` must match `fragments` length when provided.

#### waters

`waters` is a two-element array identifying the first and last water molecule to be treated classically (used by some QMMM/Q4ML workflows).

(residues)=
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

`residues` is typically aligned one-to-one with `topologies` (same index). It is required for QMMM and other workflows that rely on residues, and it is often generated by converting PDB/TRC inputs.
