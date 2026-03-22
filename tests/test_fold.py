from pathlib import Path

from rush.boltz import (
    LigandSequence,
    ProteinSequence,
    Result,
)
from rush.client import RunOpts, RunSpec, set_opts
from rush import boltz, mmseqs2


def test_fold():
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    protein_seq = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"
    mmseqs2_ref = mmseqs2.search(
        [protein_seq],
        run_opts=RunOpts(
            name="Rush-Py Test: Fold 01 Step 1 (MMseqs2)",
            tags=["rush-py", "test", "mmseqs2"],
        ),
    ).collect()
    saved_msas = mmseqs2_ref.save()
    assert saved_msas.a3m_files[0].suffix == ".a3m"
    # Pass raw object store ref to boltz ProteinSequence
    msa_obj = {"path": str(mmseqs2_ref.outputs[0].path)}
    ref = boltz.fold(
        [
            ProteinSequence(["A"], protein_seq, msa_obj),
            LigandSequence(["E"], "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
        ],
        affinity_binder_chain_id="E",
        run_opts=RunOpts(
            name="Rush-Py Test: Fold 01 Step 2 (Boltz)",
            tags=["rush-py", "test", "boltz"],
        ),
        run_spec=RunSpec(target="Bullet2", gpus=1),
    ).collect()
    output = list(ref.fetch())
    # One diffusion sample by default
    assert len(output) == 1
    assert isinstance(output[0], Result)


if __name__ == "__main__":
    test_fold()
