import sys
from pathlib import Path
from pprint import pp

from rush import from_json
from rush.auto3d import Auto3DOptions, auto3d
from rush.client import RunOpts, save_object, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    res = auto3d(
        ["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "COOH"],
        Auto3DOptions(k=5),
        run_opts=RunOpts(
            name="Rush-Py Test Auto3D 01",
            tags=["rush-py", "test"],
        ),
        collect=True,
    )
    # Output is a list of TRC objects in memory, or a str if auto3d failed
    trc_obj, err = res
    trc = from_json(tuple(save_object(o["path"]) for o in trc_obj))
    pp(trc, width=130, compact=True, stream=sys.stderr)
    print(err, file=sys.stderr)
