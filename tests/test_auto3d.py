
# Run test
import sys

from rush_py2.auto3d import auto3d
from rush_py2.client import download_object


if __name__ == "__main__":
    result = auto3d(["C1=CC=CC=C1"])
    if result is not None:
        for [smi, topology_data_as_vobj] in result["result"]:
            print(smi)
            print(object)
            topology_data_as_json = download_object(topology_data_as_vobj["path"])
            print(topology_data_as_json)
    else:
        print("No result")