import json
import sys
from pathlib import Path
from pprint import pp

from rush_py2 import from_json
from rush_py2.boltz import LigandSequence, ProteinSequence, boltz
from rush_py2.client import RunOpts, RunSpec, save_object, set_opts
from rush_py2.convert.pdb import to_pdb
from rush_py2.mmseqs2 import mmseqs2

if __name__ == "__main__":
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
    protein_seq = "LSALNPELVQAVQHVVIGPSSLIVHFNEVIGRGHFGCVYHGTLLDNDGKKIHCAVKSLNRITDIGEVSQFLTEGIIMKDFSHPNVLSLLGICLRSEGSPLVVLPYMKHGDLRNFIRNETHNPTVKDLIGFGLQVAKGMKYLASKKFVHRDLAARNCMLDEKFTVKVADFGLARDMYDKEYYSVHNKTGAKLPVKWMALESLQTQKFTTKSDVWSFGVLLWELMTRGAPPYPDVNTFDITVYLLQGRRLLQPEYCPDPLYEVMLKCWHPKAEMRPSFSELVSRISAIFSTFIG"
    res = mmseqs2(
        [protein_seq],
        collect=True,
    )
    print(res, file=sys.stderr)
    res = boltz(
        [
            ProteinSequence(["A"], protein_seq, res[0]),
        ],
        use_potentials=True,
        template_path=data_dir / "CMET_4R1Y.pdb",
        template_threshold_angstroms=0.1,
        run_opts=RunOpts(
            name="Rush-Py Test: Residue Inpainting",
            tags=["rush-py", "test", "mmseqs2", "CMET", "4r1y"],
        ),
        run_spec=RunSpec(target="Bullet2", gpus=1),
        collect=True,
    )
    trc_obj = res[0][0]
    trc = from_json(
        (
            save_object(trc_obj[0]["path"]),
            save_object(trc_obj[1]["path"]),
            save_object(trc_obj[2]["path"]),
        )
    )
    if isinstance(trc, list):
        trc = trc[0]
    (Path.cwd() / "CMET_4R1Y_inpainted.pdb").write_text(to_pdb(trc))
