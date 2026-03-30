from rush import boltz, mmseqs2
from rush.runs import RunOpts, RunSpec
from tests._module_test_utils import assert_run_collects_and_caches


def test_fold():
    protein_seq = (
        "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGEL"
        "ARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAW"
        "RNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGC"
        "GTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIF"
        "HLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEE"
        "WFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"
    )
    msa_run = mmseqs2.search(
        [protein_seq],
        run_opts=RunOpts(
            name="Rush-Py Test: Fold 01 Step 1 (MMseqs2)",
            tags=["rush-py", "test", "mmseqs2"],
        ),
    )
    msa_ref = assert_run_collects_and_caches(msa_run, mmseqs2.ResultRef)
    fetched_msas = msa_run.fetch()
    assert len(fetched_msas) == 1
    assert fetched_msas[0].startswith(">")
    saved_msas = msa_run.save()
    assert len(msa_ref) == 1
    assert saved_msas[0].suffix == ".a3m"
    assert saved_msas[0].exists()

    # Pass raw object store ref to boltz ProteinSequence
    fold_run = boltz.fold(
        [
            boltz.ProteinSequence(["A"], protein_seq, msa_ref[0]),
            boltz.LigandSequence(["E"], "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
        ],
        affinity_binder_chain_id="E",
        run_opts=RunOpts(
            name="Rush-Py Test: Fold 01 Step 2 (Boltz)",
            tags=["rush-py", "test", "boltz"],
        ),
        run_spec=RunSpec(target="Bullet2", gpus=1),
    )
    assert_run_collects_and_caches(fold_run, boltz.ResultRef)
    output = list(fold_run.fetch())
    # One diffusion sample by default
    assert len(output) == 1
    assert isinstance(output[0], boltz.Result)
    assert output[0].plddt.size > 0
    assert output[0].pae.size > 0
    assert output[0].affinities is not None

    saved = list(fold_run.save())
    assert len(saved) == 1
    assert isinstance(saved[0], boltz.ResultPaths)
    assert saved[0].model.topology.exists()
    assert saved[0].model.residues.exists()
    assert saved[0].model.chains.exists()
    assert saved[0].metrics.exists()
    assert saved[0].plddt.exists()
    assert saved[0].pae.exists()
    assert saved[0].affinities is not None
    assert saved[0].affinities.exists()
