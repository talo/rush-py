from pathlib import Path
from pprint import pp

from rush_py2 import from_json
from rush_py2.auto3d import Auto3DOptions, auto3d

if __name__ == "__main__":
    result1, result2 = auto3d(
        ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "COOH"], Auto3DOptions(k=5), collect=True
    )
    pp(from_json(result1), width=130, compact=True)
    print(result2)
