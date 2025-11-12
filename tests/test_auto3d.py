import sys
from pathlib import Path
from pprint import pp

from rush_py2 import from_json
from rush_py2.auto3d import Auto3DOptions, auto3d
from rush_py2.client import save_object, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    (t, r, c), err = auto3d(
        ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "COOH"],
        Auto3DOptions(k=5),
        collect=True,
    )
    pp(
        from_json((save_object(t), save_object(r), save_object(c))),
        width=130,
        compact=True,
        stream=sys.stderr,
    )
    print(err, file=sys.stderr)
