from pathlib import Path

from rush import boltz, mmseqs2
from rush.client import RunOpts, RunSpec
from tests._module_test_utils import assert_run_collects_and_caches


def test_fold_inpainting(test_data_dir: Path):
    protein_seq = (
        "LSALNPELVQAVQHVVIGPSSLIVHFNEVIGRGHFGCVYHGTLLDNDGKKIHCAVKSLNRITDIGEVS"
        "QFLTEGIIMKDFSHPNVLSLLGICLRSEGSPLVVLPYMKHGDLRNFIRNETHNPTVKDLIGFGLQVAKG"
        "MKYLASKKFVHRDLAARNCMLDEKFTVKVADFGLARDMYDKEYYSVHNKTGAKLPVKWMALESLQTQKF"
        "TTKSDVWSFGVLLWELMTRGAPPYPDVNTFDITVYLLQGRRLLQPEYCPDPLYEVMLKCWHPKAEMRPS"
        "FSELVSRISAIFSTFIG"
    )
    msa_run = mmseqs2.search(
        [protein_seq],
        run_opts=RunOpts(
            name="Rush-Py Test: Residue Inpainting Step 1 (MMseqs2)",
            tags=["rush-py", "test", "mmseqs2", "CMET", "4r1y"],
        ),
    )
    msa_ref = assert_run_collects_and_caches(msa_run, mmseqs2.ResultRef)
    fetched_msas = msa_run.fetch()
    assert len(fetched_msas) == 1
    assert fetched_msas[0].startswith(">")
    saved_msas = msa_run.save()
    assert saved_msas[0].suffix == ".a3m"
    assert saved_msas[0].exists()

    fold_run = boltz.fold(
        [
            boltz.ProteinSequence(["A"], protein_seq, msa_ref[0]),
        ],
        use_potentials=True,
        template_path=test_data_dir / "4r1y_protein.pdb",
        template_threshold_angstroms=0.1,
        run_opts=RunOpts(
            name="Rush-Py Test: Residue Inpainting Step 2 (Boltz)",
            tags=["rush-py", "test", "boltz", "CMET", "4r1y"],
        ),
        run_spec=RunSpec(target="Bullet", gpus=1),
    )
    assert_run_collects_and_caches(fold_run, boltz.ResultRef)

    fetched = list(fold_run.fetch())
    assert len(fetched) == 1
    assert fetched[0].plddt.size > 0
    assert fetched[0].pae.size > 0

    output = list(fold_run.save())
    # One diffusion sample by default
    assert len(output) == 1
    assert isinstance(output[0], boltz.ResultPaths)
    assert output[0].model.topology.exists()
    assert output[0].model.residues.exists()
    assert output[0].model.chains.exists()
    assert output[0].metrics.exists()
    assert output[0].plddt.exists()
    assert output[0].pae.exists()
