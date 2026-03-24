import json
import sys
from typing import Any

from rush import TRC, to_dict
from rush.convert import from_json, from_sdf
from rush import merge_trcs


def test_merge_with_paths(test_data_dir):
    """Test merging TRCs using file paths."""
    seqs = merge_trcs(
        test_data_dir / "1hsg_MK1_trc.json", test_data_dir / "1hsg_HOH_trc.json"
    ).residues.seqs
    assert "MK1" in seqs and "HOH" in seqs


def test_merge_with_trcs(test_data_dir):
    """Test merging TRCs using TRC objects."""
    mk1_trcs = from_json(test_data_dir / "1hsg_MK1_trc.json")
    assert isinstance(mk1_trcs, list)
    assert len(mk1_trcs) == 1
    mk1_trc = mk1_trcs[0]
    hoh_trcs = from_json(test_data_dir / "1hsg_HOH_trc.json")
    assert isinstance(hoh_trcs, list)
    assert len(hoh_trcs) == 1
    hoh_trc = hoh_trcs[0]
    seqs = merge_trcs(mk1_trc, hoh_trc).residues.seqs
    assert "MK1" in seqs and "HOH" in seqs


def normalize_json(obj):
    """Recursively normalize JSON to ignore key ordering."""
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    else:
        return obj


def test_merge_3fly(test_data_dir):
    """Test merging 3fly ligand SDF with 3fly protein JSON to get 3fly complex JSON."""
    protein_json_file = test_data_dir / "3fly_protein_trc.json"
    ligand_sdf_file = test_data_dir / "3fly_ligand.sdf"
    expected_complex_json_file = test_data_dir / "3fly_complex_trc.json"

    # Check that all files exist
    assert ligand_sdf_file.exists(), f"Ligand SDF file not found: {ligand_sdf_file}"
    assert protein_json_file.exists(), (
        f"Protein JSON file not found: {protein_json_file}"
    )
    assert expected_complex_json_file.exists(), (
        f"Expected complex JSON file not found: {expected_complex_json_file}"
    )

    # Load ligand from SDF
    with open(ligand_sdf_file, "r") as f:
        ligand_sdf_content = f.read()

    ligand_trc = from_sdf(ligand_sdf_content)
    assert isinstance(ligand_trc, TRC), f"Expected 1 ligand TRC, got {len(ligand_trc)}"

    # Load protein from JSON
    protein_trcs = from_json(protein_json_file)
    assert isinstance(protein_trcs, list)
    assert len(protein_trcs) == 1, f"Expected 1 protein TRC, got {len(protein_trcs)}"
    protein_trc = protein_trcs[0]

    # Merge protein and ligand
    merged_trc = merge_trcs(protein_trc, ligand_trc)

    # Convert merged TRC to JSON
    merged_json = to_dict([merged_trc])
    # to_dict returns an array, but expected might be a single object
    if isinstance(merged_json, list) and len(merged_json) == 1:
        merged_json = merged_json[0]

    # Read expected JSON
    with open(expected_complex_json_file, "r") as f:
        expected_json = json.load(f)
    # Handle both single object and array
    if isinstance(expected_json, list) and len(expected_json) == 1:
        expected_json = expected_json[0]

    # Normalize both JSON objects (to ignore key ordering)
    normalized_merged = normalize_json(merged_json)
    normalized_expected = normalize_json(expected_json)

    def summarize_required_fields(trc_obj: Any) -> dict[str, Any]:
        if isinstance(trc_obj, list):
            if not trc_obj:
                return {}
            trc_obj = trc_obj[0]

        # Extract only the fields we care about for equality.
        topology = trc_obj.get("topology", {})
        residues = trc_obj.get("residues", {})
        chains = trc_obj.get("chains", {})

        return {
            "topology": {
                "symbols_count": len(topology.get("symbols", [])),
                "connectivity_count": len(topology.get("connectivity", [])),
                "fragment_formal_charges_count": len(
                    topology.get("fragment_formal_charges", []) or []
                ),
            },
            "residues": {
                "count": len(residues.get("residues", [])),
                "labels": residues.get("labels"),
                "labeled": residues.get("labeled"),
            },
            "chains": {
                "count": len(chains.get("chains", [])),
                "labels": chains.get("labels"),
            },
        }

    merged_summary = summarize_required_fields(normalized_merged)
    expected_summary = summarize_required_fields(normalized_expected)

    if merged_summary != expected_summary:
        print(f"  Merged summary:   {merged_summary}", file=sys.stderr)
        print(f"  Expected summary: {expected_summary}", file=sys.stderr)
    assert merged_summary == expected_summary, (
        "Fields differ between merged TRC and expected complex JSON"
    )
