import json
import sys
from pathlib import Path

from rush_py2.client import RunOpts, download_object, save_json, set_opts
from rush_py2.prepare_protein import prepare_protein

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
    res = prepare_protein(
        data_dir / "cdk2_dry_trc.json",
        run_opts=RunOpts(
            name="Test prepare-protein 01", tags=["rush-py2", "test", "cdk2"]
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    t_dict = json.loads(download_object(res[0]["path"]).decode())
    r_dict = json.loads(download_object(res[1]["path"]).decode())
    c_dict = json.loads(download_object(res[2]["path"]).decode())
    trc_o_dict = {"topology": t_dict, "residues": r_dict, "chains": c_dict}
    trc_name = f"{res[0]['path'][:8]}_{res[1]['path'][:8]}_{res[2]['path'][:8]}"
    save_json(trc_o_dict, name=trc_name)
