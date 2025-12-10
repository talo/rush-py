import json
import sys
from pathlib import Path

from rush_py2.client import RunOpts, save_json, save_object, set_opts
from rush_py2.convert import from_json, to_json
from rush_py2.prepare_protein import prepare_protein

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
    res = prepare_protein(
        data_dir / "1hsg_trc.json",
        ph=7.0,
        naming_scheme="CHARMM",
        capping_style="always",
        truncation_threshold=5,
        run_opts=RunOpts(
            name="Test prepare-protein 01", tags=["rush-py2", "test", "cdk2"]
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    trc = from_json(
        (
            save_object(res[0]["path"]),
            save_object(res[1]["path"]),
            save_object(res[2]["path"]),
        )
    )
    trc_name = f"{res[0]['path'][:8]}_{res[1]['path'][:8]}_{res[2]['path'][:8]}"
    save_json(json.loads(to_json(trc)), name=trc_name)
