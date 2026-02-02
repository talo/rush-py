# Exporting Data from EXESS

EXESS provides its primary output through a JSON file, which through rush-py is its first (zeroth-indexed) output. Additional data can be requested through the `export_keywords` parameter.


## Example: Obtaining the Output Files

After calling EXESS with this parameter set:
```python
from rush import exess
from rush.client import RunOpts
res = exess.energy(
    "input_topology.json",
    export_keywords=exess.ExportKeywords(
        export_density=True,  # Export electron density values
    ),
    run_opts=RunOpts(
        name="Rush-Py Tutorial: EXESS Exports 1",
        tags=["rush-py", "tutorial", "exess"],
    ),
    collect=True,
)
```

We can inspect the outputs by printing `res`, which might look something like this:
```python
[
    {'path': '17e16a82-b659-4fc7-83d1-740f65cfee17', 'size': 0, 'format': 'json'},
    {'path': '3f80961e-173b-44d9-9462-7a7e201a7394', 'size': 0, 'format': 'bin'}
]
```

The path tells us a unique ID associated with this output, and we can use this path to download the object to our filesystem. The size is currently uninformative, but will eventually contain that output's actual size in bytes. The format tells us whether the data is directly downloadable as JSON, or whether it's in another format, treated by the Rush object storage as effectively binary data.

The `exess.energy` function has an associated `save_energy_outputs` function that will do the legwork for us to download these output files. We just have to pass it `res`:
```python 
files = exess.save_energy_outputs(res)
```

And now in `files` we have the on-disk paths to the downloaded versions of these files.

The `exess.save_energy_outputs` call saves the first output as a JSON file into our workspace, using the above object store path as the filename.

If any exports are requested, it also saves the second output to the workspace with an `.hdf5` extension (after all, it is an HDF5 file). The helper handles the format and extension automatically. The HDF5 file contains these exported values. In the above case, we're exporting the electron density of the system. The exported data can be quite large, hence the usage of HDF5 files rather than plain old JSON.

Feel free to use h5py, CLI programs like `h5glance`, `h5ls`, and `h5dump`, or your preferred tool for working with the exported output data.


## Example: Descriptor Grids for Electron Density and Electrostatic Potential

There are two exportable values that are "descriptors": `density_descriptors` and `esp_descriptors`. (NOTE: `expanded_esp_descriptors` currently crashes with an internally determined OOM error, so please don't enable it.) When these are used, the `descriptor_grid` keyword must also be set. This keyword defines a grid upon which the values are calculated, allowing the user to obtain values at the desired coarseness or fineness or even at exact points. For example, the following code will collect the electron density and electrostatic potential (ESP) descriptors in a rectangular region bounded by min points `(0.0, 0.0, 0.0)` and `(1.0, 1.0, 1.0)`, and with spacing of 1 Angstrom in each dimension. So, our grid will be the eight vertices of a cube.
```python
from rush import exess
from rush.client import RunOpts, RunSpec
res = exess.energy(
    "input_topology.json",
    frag_keywords=None,  # No fragmentation; whole system calc
    export_keywords=exess.ExportKeywords(
        export_density_descriptors=True,
        export_esp_descriptors=True,
        descriptor_grid=exess.RegularDescriptorGrid(
            min=[0.0, 0.0, 0.0],
            max=[1.0, 1.0, 1.0],
            spacing=[1.0, 1.0, 1.0],
        ),
    ),
    convert_hdf5_to_json=True,
    run_spec=RunSpec(storage=1000, gpus=1),
    run_opts=RunOpts(
        name="Rush-Py Tutorial: EXESS Exports 2",
        tags=["rush-py", "tutorial", "exess", "electron density", "ESP"],
    ),
    collect=True,
)
```

Note that we did not use fragmentation, which complicates the interpretation of the descriptor grid values.

Because the descriptor grid values aren't particularly numerous, it may be more convenient to save them as JSON. This can be done by passing `convert_hdf5_to_json=True` to exess.energy as shown above and then saving the output using `exess.save_energy_outputs`:
```python
files = exess.save_energy_outputs(res)
```

After opening the saved JSON file, one might see output as follows:
```json
{
  "density_descriptors": [
    0.019335589097923125,
    0.006869915843364948,
    0.0005634210968506005,
    0.2633564534732077,
    0.03143799260181341,
    0.0012500707523984667,
    0.05346373829046323,
    0.010633021816107797,
  ],
  "esp_descriptors": [
    -14.738774776839207,
    -12.25448503958538,
    -8.832058509543936,
    -15.686691210669682,
    -12.001374325303633,
    -8.482681332376423,
    -11.792483297622633,
    -9.824425520888148,
  ],

  "descriptor_grid": [
    [ 0.0, 0.0, 0.0 ],
    [ 0.0, 0.0, 1.0 ],
    [ 0.0, 1.0, 0.0 ],
    [ 0.0, 1.0, 1.0 ],
    [ 1.0, 0.0, 0.0 ]
    [ 1.0, 0.0, 1.0 ]
    [ 1.0, 1.0, 0.0 ]
    [ 1.0, 1.0, 1.0 ]
  ],
  "descriptor_grid_weights": [
    1.0,
    1.0,
    1.0
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ]
}
```

The data for the exported values are present, and the descriptor grid coordinates are given as well. Descriptor grid weights are also present.

This will convert the HDF5 file to JSON and save it to the workspace with its filename as the path of the original HDF5 output, same as if the HDF5 file itself had been saved, but with the `.json` extension instead, as expected.

Note that in the case of fragmentation, the data are stored on a per-nmer basis, and if both descriptors are requested, they will be stored together at the innermost level of nesting in the HDF5 file. When converted to JSON, the different descriptors are stored in entirely separate outer keys in the JSON file, making it easy to work with just one or the other. Otherwise, the structure of the HDF5 file is preserved in the JSON file.

### Other Types of Descriptor Grids

In addition to the `RectangularDescriptorGrid` shown in the example, a descriptor grid can also be constructed via `CustomDescriptorGrid`, `StandardDescriptorGrid` or `DescriptorGrid`. 

The `CustomDescriptorGrid` can be constructed by passing a single list with the x, y, and z coordinates of each point in the grid listed sequentially, i.e. `[x1, y1, z1, x2, y2, z2, ...]`. Only small (i.e., less than hundreds) numbers of points should be specified using this method.

The `StandardDescriptorGrid` takes a single string keyword. Valud values are `"FINE"`, `"ULTRAFINE"`, `"SUPERFINE"`, `"TREUTLER_GM3"`, and `"TREUTLER_GM5"`. These are radial grids centered around the atom locations. `StandardDescriptorGrid("ULTRAFINE")` can serve as a good baseline if a radial grid is desired.

The `DescriptorGrid` class takes three arguments to its constructor: `points_per_shell` (an int), `order` (must be either `"One"` or `"Two"`), and `scale` (a float).
