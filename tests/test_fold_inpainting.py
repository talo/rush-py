from pathlib import Path

from rush.boltz import ProteinSequence, ResultPaths
from rush.client import RunOpts, RunSpec, set_opts
from rush import boltz, mmseqs2


def test_fold_inpainting():
    set_opts(workspace_dir=Path.cwd() / ".scratch" / "workspace")
    data_dir = Path.cwd() / "tests" / "data"
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
    msa_obj = {"path": str(mmseqs2_ref[0].path)}
    ref = boltz.fold(
        [
            ProteinSequence(["A"], protein_seq, msa_obj),
        ],
        use_potentials=True,
        template_path=data_dir / "4r1y_protein.pdb",
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


if __name__ == "__main__":
    test_fold_inpainting()
