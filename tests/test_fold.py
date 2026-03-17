import json
import sys
from pathlib import Path

from rush.boltz import (
    BoltzResult,
    LigandSequence,
    ProteinSequence,
    boltz,
    fetch_outputs,
)
from rush.client import RunError, RunOpts, RunSpec, set_opts
from rush.mmseqs2 import mmseqs2
from rush.mmseqs2 import save_outputs as save_mmseqs2_outputs


def test_fold():
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    protein_seq = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"
    res = mmseqs2(
        [protein_seq],
        run_opts=RunOpts(
            name="Rush-Py Test: Fold 01 Step 1 (MMseqs2)",
            tags=["rush-py", "test", "mmseqs2"],
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
            LigandSequence(["E"], "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
        ],
        affinity_binder_chain_id="E",
        run_opts=RunOpts(
            name="Rush-Py Test: Fold 01 Step 2 (Boltz)",
            tags=["rush-py", "test", "boltz"],
        ),
        run_spec=RunSpec(target="Bullet2", gpus=1),
        collect=True,
    )
    assert not isinstance(res, (str, RunError))
    print(json.dumps(res, indent=2), file=sys.stderr)
    output = fetch_outputs(res)
    assert not isinstance(output, (str, RunError))
    assert isinstance(output[0], BoltzResult)


if __name__ == "__main__":
    test_fold()
