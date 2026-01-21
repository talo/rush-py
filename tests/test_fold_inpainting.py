import json
import sys
from pathlib import Path

from rush.boltz import ProteinSequence, boltz
from rush.client import RunOpts, RunSpec, set_opts
from rush.mmseqs2 import mmseqs2


def test_fold_inpainting():
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
    protein_seq = "LSALNPELVQAVQHVVIGPSSLIVHFNEVIGRGHFGCVYHGTLLDNDGKKIHCAVKSLNRITDIGEVSQFLTEGIIMKDFSHPNVLSLLGICLRSEGSPLVVLPYMKHGDLRNFIRNETHNPTVKDLIGFGLQVAKGMKYLASKKFVHRDLAARNCMLDEKFTVKVADFGLARDMYDKEYYSVHNKTGAKLPVKWMALESLQTQKFTTKSDVWSFGVLLWELMTRGAPPYPDVNTFDITVYLLQGRRLLQPEYCPDPLYEVMLKCWHPKAEMRPSFSELVSRISAIFSTFIG"
    res = mmseqs2(
        [protein_seq],
        run_opts=RunOpts(
            name="Rush-Py Test: Residue Inpainting Step 1 (MMseqs2)",
            tags=["rush-py", "test", "mmseqs2", "CMET", "4r1y"],
        ),
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
            name="Rush-Py Test: Residue Inpainting Step 2 (Boltz)",
            tags=["rush-py", "test", "boltz", "CMET", "4r1y"],
        ),
        run_spec=RunSpec(target="Bullet2", gpus=1),
        collect=True,
    )
    print(json.dumps(res, indent=2), file=sys.stderr)


if __name__ == "__main__":
    test_fold_inpainting()
