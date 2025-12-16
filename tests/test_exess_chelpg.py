import sys
from pathlib import Path
from pprint import pp

from rush_py2 import exess
from rush_py2.client import RunOpts, save_json, set_opts

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = exess.chelpg(
        data_dir / "tyk2_ejm_31_t.json",
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 03: ChelpG",
            tags=["rush-py", "test", "tyk2+ejm-31"],
        ),
        collect=True,
    )
    pp(res, width=130, compact=True, stream=sys.stderr)
    save_json(res, name="tyk2_ejm_31_chelpg_charges.json")
