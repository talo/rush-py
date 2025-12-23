import json
import sys
from pathlib import Path

from rush_py2.client import RunOpts, save_object, set_opts
from rush_py2.nnxtb import NnxtbResults, nnxtb

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = nnxtb(
        data_dir / "1kuw_t.json",
        compute_forces=True,
        compute_frequencies=False,  # defaults to False; more expensive to compute
        multiplicity=1,  # also the default (singlet)
        run_opts=RunOpts(
            name="Rush-Py Test NN-xTB 01",
            tags=["rush-py2", "test", "1kuw"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    print(
        NnxtbResults(**json.loads(save_object(res["path"]).read_text())),
        file=sys.stderr,
    )
