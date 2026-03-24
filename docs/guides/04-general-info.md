# General Info and Design

The rush-py client functions have a structured form for their arguments:
- The first set of arguments are the input data;
- The remaining arguments are the configuration values;
- When large inputs and outputs are required, paths to objects in the Rush object store are used instead of the in-memory values.

To see the documentation, signature, or parameters for any class or function, use Python's built-in help function:
```python
from rush import exess

help(exess.energy)
help(exess.FragKeywords)
```

## Automatic File Conversion
When a Rush module expects a tuple of paths to Topology, Residues, and Chains objects to represent a molecular system, rush-py will alternatively accept a Path object to a PDB file or SDF file for proteins and ligands respectively, or a path to a TRC file on disk.

## Run Options (Metadata)
One can pass a set of run options to each module function via `run_opts=rush.client.RunOpts(...)`. Current options include setting the run's name, description, tags, and an email flag which, if set to true, will trigger messages for job notifications sent to the email address associated with the user's Rush account.

## Submit + Collect Pattern
Rush module functions return a `RushRun` handle. Call `RushRun.collect()` to wait for completion and get the module's result reference, or use the convenience shortcuts `RushRun.fetch()` and `RushRun.save()`. All functions take a `max_wait_time` parameter that sets the number of seconds to wait for the run to finish before timing out.

## Uploading, Downloading & Saving Data
The Rush client module provides `client.upload_object` and `client.save_object`, which allow for uploading and saving `RushObject` instances to the Rush object store to and from local filesystem paths. Also, each module's `ResultRef` class provides `ResultRef.fetch()` and `ResultRef.save() functions. These fetch a module's results and return its data directly in memory, and save an object into the workspace directory with arguments that allow for configuring how it gets named.

Also provided is `client.save_json`, which allows saving a dict as JSON, by default into the workspace directory, for convenient parallel usage with `save_object`.

Downloading outputs is not required when chaining module runs: the object store paths returned by a module can be passed directly into another module as inputs.

## Workspaces
Workspaces are used to organize output files from Rush runs. When using `RushRun.save()` or `ResultRef.save()`, a folder is created for the project currently in use named via the project ID, and the files are saved based on their object store paths. In this way, the output files will never be overwritten, as object store paths are guaranteed to be unique.

A `history.json` file is also written into the root of the workspace, where it maintains a list of all module instances (i.e. runs of a module) that have been created for this workspace. Each module instance has its run ID, time created, and module path (which contains the exact revision of the module used for the run) tracked here.

There is no need to manually create a "client" to submit jobs to Rush, but you can configure the workspace by importing rush.client:

```python
from rush import client
client.set_opts(workspace_dir=Path("/path/to/desired/workspace/folder/"))
```

## Opening Issues

If you find that any Rush module's Python submodule doesn't abide by these design criteria or behaves in an unexpected way, please file a bug report by [opening an issue](https://github.com/talo/rush-py/issues/new)!

## Submitting Bug Reports and Feedback

If you want to submit bug reports or give us any feedback for any part of Rush, you can use [this form](https://forms.gle/1DPWK91utzJ6SED47) as well!
