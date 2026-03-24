import json
from pathlib import Path

import pytest

from rush.convert import from_pdb, from_sdf, to_dict
from rush.fried import fragment_ligand, fragmented_exess, plot_fried_stacked
from rush import merge_trcs


@pytest.mark.timeout(2700)
def test_fried_3fln(tmp_path: Path, test_data_dir: Path):
    system_name = "3fln"
    protein_trc = from_pdb((test_data_dir / f"{system_name}_protein.pdb").read_text())
    ligand_trc = from_sdf((test_data_dir / f"{system_name}_ligand.sdf").read_text())

    complex_trc = merge_trcs(protein_trc, ligand_trc)
    complex_json_path = tmp_path / f"{system_name}_complex.json"
    with complex_json_path.open("w") as f:
        json.dump(to_dict(complex_trc), f, indent=2)

    fragmented_lig_file = fragment_ligand(complex_json_path)
    fragmented_exess(fragmented_lig_file, distance_threshold=3)

    plot_fried_stacked(tmp_path, system_prefix=f"{system_name}_complex")
