from pathlib import Path

from rush.client import set_opts
from rush.convert import from_pdb, from_sdf, to_json
from rush.fried import fragment_ligand, fragmented_exess, plot_fried_stacked
from rush.trc.merge import merge_trcs


def test_fried_3fln(tmp_path: Path):
    set_opts(workspace_dir=tmp_path / ".test-workspace")
    data_dir = Path(__file__).parent / "data"

    system_name = "3fln"
    protein_trc = from_pdb((data_dir / f"{system_name}_protein.pdb").read_text())
    ligand_trc = from_sdf((data_dir / f"{system_name}_ligand.sdf").read_text())

    complex_trc = merge_trcs(protein_trc, ligand_trc)
    complex_json_path = tmp_path / f"{system_name}_complex.json"
    complex_json_path.write_text(to_json(complex_trc))

    fragmented_lig_file = fragment_ligand(complex_json_path)
    fragmented_exess(fragmented_lig_file, distance_threshold=3)

    plot_fried_stacked(tmp_path, system_prefix=f"{system_name}_complex")


if __name__ == "__main__":
    test_fried_3fln(Path.cwd())
