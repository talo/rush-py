# Rush-Py Tutorial

Rush consists of modules, which are individual programs or entrypoints for running programs. Rush-py provides a function for each module that allows you to call it, and supports the full set of inputs and outputs for each module (in most cases). The design stays as faithful as possible to the module specification itself, only deviating to make the modules easier to use.

## Rush-Py Design and Usage

### Access

Use environment variables to configure access:
- `RUSH_TOKEN`: Put your token's value here
- `RUSH_PROJECT`: Put your project's UUID value here; can find it in the URL once selecting a project in the Rush UI
- `RUSH_ENDPOINT`: Use this to choose between staging and prod; if omitted, defaults to prod

### Basics

Each Rush module has a Python submodule that provides support for it. For example, to get access to the EXESS module support, you can import files and functions from `rush.exess`:
```python
from rush import exess
```

You can then access the function that directly wraps EXESS as expected:
```python
from rush import exess
exess.exess("input_topology.json", <rest of args here>)
```

Or the possibly more clear:
```python
from rush.exess import exess as run_exess
run_exess("input_topology.json", <rest of args here>)
```

Or access one of the other supplementary EXESS entrypoints, built to facilitate easily performing different kinds of runs with EXESS:
```python
from rush import exess
exess.energy(...)
exess.interaction_energy(...)
exess.chelpg(...)
exess.optimization(...)
exess.qmmm(...)
```

To see the documentation, signature, or parameters for any class or function, you can write:
```python
help(exess.energy)
help(exess.FragKeywords)
```

If you find that any Rush module's python submodule doesn't abide by these design criteria or behaves in an unexpected way, please file a bug report!

### Configuring Rush-Py's Behavior

Unlike in the classic rush-py library, there is no need to manually create a "client" to submit jobs to Rush. But, there are some configurable behaviors that can be accessed by importing `rush.client`:
```python
from rush import client
client.set_opts(workspace_dir=Path("/path/to/desired/workspace/folder/"))
```

As of now, the workspace directory is the only configurable value.

### Workspaces

Workspaces are used to organize output files from Rush runs. When using the `save_outputs` functions as described above, a folder is created for the project currently in use named via the project ID, and the files are saved based on their object store paths. In this way, the output files will never be overwritten, as object store paths are guaranteed to be unique.

A `history.json` file is also written into the root of the workspace, where it maintains a list of all module instances (i.e. runs of a module) that have been created for this workspace. Each module instance has its run ID, time created, and module path (which contains the exact revision of the module used for the run) tracked here.

### Uploading, Downloading & Saving Data

Rush-py provides an `upload_object`, `download_object`, and `save_object` function in the `rush.client` python submodule. These functions: upload to an object from a local filesystem path; download an object via its object store path and return its data directly (either as a dict for JSON data or as bytes otherwise; and saves an object into the workspace directory, with arguments that allow for configuring how it gets named (run `help(rush.client.save_object)` for usage).

Also provided is a `save_json` function that allows saving a dict as JSON, by default into the workspace directory, for convenient parallel usage with `save_object`.

### Run Options (Metadata)

One can pass a set of run options to each module function via `run_opts=rush.client.RunOpts(...)`. Current options include setting the run's name, description, tags, and an email flag which if set to true will trigger messages for job notifications sent to the email address associated with the user's Rush account.

When necessary, default `RunSpec` objects are set for module functions that require the use of GPUs. Be careful when providing your own `RunSpec` such that you use supported targets for the module being run and supply at least the minimum required resources!

### Run Specification (Target + Resources)

Obe can pass a set of run specifications to each module function via `run_spec=rush.client.RunSpec(...)`. The target (Bullet, Bullet2, Bullet3, Gadi, Setonix), walltime (in minutes), storage (in MB, though storage units are configurable as well), cpus, gpus, and nodes are all configurable via this parameter and class.

### Submit + Collect Pattern

Rush modules functions will return a run ID that can be used to collect the run at a later point in time using `rush.client.collect_run`, which takes the run ID and a maximum time to wait for the run to finish (1 hour by default). If it does finish, the `collect_run` call will then return the module outputs.

If synchronous behavior is desired, `collect=True` can be passed to the module function and collection with a 1 hour wait time will happen automatically, without the need to call `collect_run`, and the module outputs will be returned directly from the module function call as well.

### Utilities 

Rush-py provides some additional utilities alongside the modules.

#### Output Saving Helpers

Some module functions have a `save_outputs` function associated with them that will automatically download the objects to the local filesystem and return the paths to those downloaded files instead of the paths to the objects in the Rush store. Note that downloading objects isn't necessary if the outputs are going to be used as inputs to another module - the object store paths can be passed directly as input to the next module. The `save_outputs` function is designed to retain the same output signature as the main module and its function, but with the object store paths transformed into local filesystem paths.

NOTE: This design hasn't been implemented for each module yet. Please file an issue to request them  or improvements to this design if you find them useful!

#### Working with TRCs

There are utilities for easily using TRC files:
- `from_pdb` and `to_pdb`: for converting to and from PDB files;
- `from_json` and `to_json` (and `from_json` static methods on `Topology`, `Residues`, and `Chains` objects): read and write TRC JSON files, and read separate `Topology`, `Residues`, and `Chains` files;
- `TRC` methods: `check`, `extend`, and `new_trc_from_residue_subset` for working with TRCs. The latter provides a way to split TRCs into, e.g., just the protein or ligand of interest, or to extract any smaller part of a larger system as its own system. These methods are all available on the individual T, R, and C classes as well.
- `Topology` methods: `distance_between_atoms`, `distance_to_point` (from an atom to a point in 3D space) `get_atoms_near_point`, and `get_fragments_near_fragment`. The latter is useful for selecting a region of a user-specified radius around a fragment.
- `Residues` methods: `is_amino_acid` checks whether a residue at a given index is an amino acid.


## Rush Module design

Rush modules have a structured form for their arguments:
- The first argument is a configuration object;
- The second and remaining objects are input data passed into the module;
- When large inputs and outputs are required, paths to objects in an object store are used instead of the objects themselves.

### Differences between rush-py and upstream modules

#### Argument order

To make the modules easier to use, rush-py rearranges this such that the first set of arguments is the input data, and the remaining arguments are the expanded configuration values.

#### Automatic file conversion

When a Rush module expects a tuple of paths to Topology, Residues, and Chains objects to represent a molecular system, rush-py will accept a `Path` object to a PDB file or SDF file for proteins and ligands respectively, or a path to a TRC file on disk, in addition to a tuple of paths to the indidvidual TRC object store path tuple that the module would normally expect.
