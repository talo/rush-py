# Objects & TRC Files

Rush has its own way of managing large files, most of which are files storing
the data for molecular systems.

## Rush Object Store

Rush stores large inputs and outputs in an object store and references them by
unique paths, which are UUIDS. These object store paths can be passed directly
between module calls, which avoids downloading and re-uploading data when
chaining runs. They can be thought of as paths the same way a path to a file on
your local computer are paths, the only difference being the need to explicitly
download or save them to your local filesystem to inspect or use the data
yourself. When objects are returned from the output of a run, they are provided
using the following JSON format:
```json
{
  "path": "UUID_OBJECT_STORE_PATH",
  "size": 0,
  "format": "bin"
}
```

The `"UUID_OBJECT_STORE_PATH"` might look something like
`"17e16a82-b659-4fc7-83d1-740f65cfee17"`. The size is currently uninformative,
but will eventually contain that output's actual size in bytes. Objects are
JSON (format `json`) or binary (format `bin`). The format tells us whether the
data is directly downloadable as JSON, or whether it's in another format,
treated by the Rush object storage as effectively binary data.

## TRC Format

TRC is a JSON-based format, built on QCSchema concepts, for representing
molecular systems. It is comparable in scope to PDB or mmCIF.

A TRC is made up of three objects:
- `Topology`: atom types, geometry, connectivity, charges, and fragments.
- `Residues`: definitions of molecular subcomponents.
- `Chains`: usually separate molecules.

A bare-bones `Topology` file without any optional fields can work well for
simple use cases, nearly as minimal as an XYZ-style geometry.

## Working with TRCs

Rush-py includes helpers for working with TRC files and their components:

- `from_pdb` and `to_pdb`: convert to and from PDB files.
- `from_json` and `to_json` (and `from_json` static methods on `Topology`,
  `Residues`, and `Chains`): read and write TRC JSON files, including the
  separate `Topology`, `Residues`, and `Chains` objects.
- `TRC` methods: `check`, `extend`, and `new_trc_from_residue_subset` for
  validating and slicing TRCs. The subset helper provides a way to split TRCs
  into, for example, just the protein or ligand of interest, or to extract any
  smaller part of a larger system as its own system. These methods are available
  on the individual `Topology`, `Residues`, and `Chains` classes as well.
- `Topology` methods: `distance_between_atoms`, `distance_to_point`,
  `get_atoms_near_point`, and `get_fragments_near_fragment`. These are useful
  for selecting a region of a user-specified radius around a fragment.
- `Residues` methods: `is_amino_acid` for residue classification.
