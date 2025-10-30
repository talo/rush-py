
# Run test
import json

from rush_py2.auto3d import auto3d
from rush_py2.client import download_object

if __name__ == "__main__":
    run = auto3d(["C1=CC=CC=C1"])
    if run is not None:
        for [smi, topology_data_as_vobj] in run["result"]:
            print(smi)
            print(object)
            topology_data_as_json = download_object(topology_data_as_vobj["path"])
            print(topology_data_as_json)
            topology_data = json.loads(topology_data_as_json)
            print(topology_data)
    else:
        print("No run available")