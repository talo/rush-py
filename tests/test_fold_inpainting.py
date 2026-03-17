import json
import sys
from pathlib import Path

from rush.boltz import BoltzSavedResult, ProteinSequence, boltz, save_outputs
from rush.client import RunError, RunOpts, RunSpec, set_opts
from rush.mmseqs2 import mmseqs2
from rush.mmseqs2 import save_outputs as save_mmseqs2_outputs


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
    assert not isinstance(res, (str, RunError))
    print(json.dumps(res, indent=2), file=sys.stderr)
    saved_msas = save_mmseqs2_outputs(res)
    assert not isinstance(saved_msas, (str, RunError))
    assert saved_msas[0].suffix == ".a3m"
    res = boltz(
        [
            ProteinSequence(["A"], protein_seq, res[0]),
        ],
        use_potentials=True,
        template_path=data_dir / "4r1y_protein.pdb",
        template_threshold_angstroms=0.1,
        run_opts=RunOpts(
            name="Rush-Py Test: Residue Inpainting Step 2 (Boltz)",
            tags=["rush-py", "test", "boltz", "CMET", "4r1y"],
        ),
        run_spec=RunSpec(target="Bullet", gpus=1),
        collect=True,
    )
    assert not isinstance(res, (str, RunError))
    print(json.dumps(res, indent=2), file=sys.stderr)
    output = save_outputs(res)
    assert not isinstance(output, (str, RunError))
    assert isinstance(output[0], BoltzSavedResult)


if __name__ == "__main__":
    test_fold_inpainting()
