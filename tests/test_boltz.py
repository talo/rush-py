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
                "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ",
                data_dir / "seq1.a3m",
            ),
            LigandSequence(["E"], "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
        ],
        affinity_binder_chain_id="E",
        collect=True,
        run_opts=RunOpts(name="Rush-Py Test: Boltz", tags=["rush-py", "test", "boltz"]),
    )
    print(res, file=sys.stderr)
