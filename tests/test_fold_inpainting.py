from pathlib import Path

from rush.boltz import ProteinSequence, ResultPaths
from rush.client import RunOpts, RunSpec
from rush import boltz, mmseqs2


def test_fold_inpainting(test_data_dir: Path):
    protein_seq = "LSALNPELVQAVQHVVIGPSSLIVHFNEVIGRGHFGCVYHGTLLDNDGKKIHCAVKSLNRITDIGEVSQFLTEGIIMKDFSHPNVLSLLGICLRSEGSPLVVLPYMKHGDLRNFIRNETHNPTVKDLIGFGLQVAKGMKYLASKKFVHRDLAARNCMLDEKFTVKVADFGLARDMYDKEYYSVHNKTGAKLPVKWMALESLQTQKFTTKSDVWSFGVLLWELMTRGAPPYPDVNTFDITVYLLQGRRLLQPEYCPDPLYEVMLKCWHPKAEMRPSFSELVSRISAIFSTFIG"
    mmseqs2_ref = mmseqs2.search(
        [protein_seq],
        run_opts=RunOpts(
            name="Rush-Py Test: Residue Inpainting Step 1 (MMseqs2)",
            tags=["rush-py", "test", "mmseqs2", "CMET", "4r1y"],
        ),
    ).collect()
    saved_msas = mmseqs2_ref.save()
    assert saved_msas[0].suffix == ".a3m"
    ref = boltz.fold(
        [
            ProteinSequence(["A"], protein_seq, mmseqs2_ref[0]),
        ],
        use_potentials=True,
        template_path=test_data_dir / "4r1y_protein.pdb",
        template_threshold_angstroms=0.1,
        run_opts=RunOpts(
            name="Rush-Py Test: Residue Inpainting Step 2 (Boltz)",
            tags=["rush-py", "test", "boltz", "CMET", "4r1y"],
        ),
        run_spec=RunSpec(target="Bullet", gpus=1),
    ).collect()
    output = list(ref.save())
    # One diffusion sample by default
    assert len(output) == 1
    assert isinstance(output[0], ResultPaths)
