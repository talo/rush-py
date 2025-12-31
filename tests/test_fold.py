import json
import sys
from pathlib import Path
from pprint import pp

from rush_py2 import from_json
from rush_py2.boltz import LigandSequence, ProteinSequence, boltz
from rush_py2.client import RunOpts, RunSpec, save_object, set_opts
from rush_py2.mmseqs2 import mmseqs2

if __name__ == "__main__":
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
    print(res, file=sys.stderr)
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
    print(json.dumps(res, indent=2), file=sys.stderr)
