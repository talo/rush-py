# rush bindings

## Usage
```python
from rush_py2 import energy, interaction_energy, chelpg, qmmm
my_topology_file = Path("path_goes_here")
energy(my_topology_file)
interaction_energy(my_topology_file)
chelpg(my_topology_file)
qmmm(my_topology_file, n_timesteps=50000)
```

## Rush setup

Use environment variables to configure access:
- `RUSH_TOKEN`: put your token's value here
- `RUSH_PROJECT`: put your project's UUID value here; can find it in the URL once
  selecting a project
