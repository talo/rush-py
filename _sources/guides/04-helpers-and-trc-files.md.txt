# Helpers and TRC Files

This guide covers helper functions and TRC-related tools that make it easier to
work with Rush outputs and molecular systems.

## Helpers

Rush-py provides helper functions alongside the module functions.

### Output Saving Helpers

Some module functions have a `save_outputs` helper that automatically downloads
objects to the local filesystem and returns local paths instead of object store
paths. This is useful when you want to inspect or post-process results on disk.

Downloading outputs is not required when chaining module runs: the object store
paths returned by a module can be passed directly into another module as inputs.

The `save_outputs` helpers are designed to retain the same output signature as
the main function, but with object store paths transformed into local filesystem
paths. Note that not every module has a `save_outputs` helper yet; if you rely
on this pattern and find a gap, please open an issue so it can be prioritized.

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
