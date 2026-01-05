# General Info and Design

The rush-py client functions have a structured form for their arguments:
- The first set of arguments are the input data;
- The remaining arguments are the configuration values;
- When large inputs and outputs are required, paths to objects should be used instead of the objects themselves.

To see the documentation, signature, or parameters for any class or function, use Python's built-in help function:
```python
help(exess.energy)
help(exess.FragKeywords)
```

## Automatic File Conversion
When a Rush module expects a tuple of paths to Topology, Residues, and Chains objects to represent a molecular system, rush-py will alternatively accept a Path object to a PDB file or SDF file for proteins and ligands respectively, or a path to a TRC file on disk.

When necessary, default RunSpec objects are set for module functions that require the use of GPUs. Be careful when providing your own RunSpec such that you use supported targets for the module being run and supply at least the minimum required resources!

## Run Options (Metadata)
One can pass a set of run options to each module function via `run_opts=rush.client.RunOpts(...)`. Current options include setting the run's name, description, tags, and an email flag which, if set to true, will trigger messages for job notifications sent to the email address associated with the user's Rush account.

## Submit + Collect Pattern
Rush module functions will return a run ID that can be used to collect the run at a later point in time using `rush.client.collect_run`, which takes the run ID and a maximum time to wait for the run to finish (1 hour by default). If it does finish, the `collect_run` call will then return the module outputs.

If synchronous behavior is desired, `collect=True` can be passed to the module function and collection with a 1 hour wait time will happen automatically, without the need to call `collect_run`, and the module outputs will be returned directly from the module function call as well.

## Uploading, Downloading & Saving Data
The rush-py client provides `upload_object`, `download_object`, and `save_object` functions in the `rush.client` Python submodule. These functions upload an object from a local filesystem path, download an object via its object store path and return its data directly (either as a dict for JSON data or as bytes otherwise), and save an object into the workspace directory with arguments that allow for configuring how it gets named (run `help(rush.client.save_object)` for usage).

Also provided is a `save_json` function that allows saving a dict as JSON, by default into the workspace directory, for convenient parallel usage with `save_object`.

## Workspaces
Workspaces are used to organize output files from Rush runs. When using the `save_outputs` functions as described above, a folder is created for the project currently in use named via the project ID, and the files are saved based on their object store paths. In this way, the output files will never be overwritten, as object store paths are guaranteed to be unique.

A `history.json` file is also written into the root of the workspace, where it maintains a list of all module instances (i.e. runs of a module) that have been created for this workspace. Each module instance has its run ID, time created, and module path (which contains the exact revision of the module used for the run) tracked here.

There is no need to manually create a "client" to submit jobs to Rush, but you can configure the workspace by importing rush.client:

```python
from rush import client
client.set_opts(workspace_dir=Path("/path/to/desired/workspace/folder/"))
```

## Opening Issues

If you find that any Rush module's Python submodule doesn't abide by these design criteria or behaves in an unexpected way, please file a bug report by [opening an issue](https://github.com/talo/rush-py/issues/new)!
