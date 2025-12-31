import json
import sys
from pathlib import Path

from rush_py2.boltz import LigandSequence, ProteinSequence, boltz
from rush_py2.client import RunOpts, RunSpec, save_object, set_opts

if __name__ == "__main__":
    data_dir = Path.cwd() / "tests" / "data"
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")

    res = boltz(
        [
            ProteinSequence(
                ["A"],
                "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL",
                data_dir / "cdk2_A.a3m",
            ),
        ],
        use_potentials=True,
        template_path=data_dir / "1b39_trc.json",
        template_threshold_angstroms=0.1,
        collect=True,
        run_opts=RunOpts(
            name="Rush-Py Test: Boltz",
            tags=["rush-py", "test", "boltz", "template"],
        ),
    )
    print(res)
