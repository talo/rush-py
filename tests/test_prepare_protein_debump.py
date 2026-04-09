import math

from rush import Element, RunOpts, from_pdb, prepare
from tests._module_test_utils import assert_run_collects_and_caches


def _as_trc(trc):
    return trc[0] if isinstance(trc, list) else trc


def _atom_label_map(trc, residue_idx, include_hydrogens=False):
    labels = trc.topology.labels
    symbols = trc.topology.symbols
    assert labels is not None, "Expected topology labels for atom matching."
    residue = trc.residues.residues[residue_idx]
    mapping = {}
    for atom_idx in residue.atoms:
        if not include_hydrogens and symbols[atom_idx] == Element.H:
            continue
        label = labels[atom_idx]
        if label in mapping:
            raise AssertionError(
                f"Duplicate atom label {label} in residue {residue_idx}."
            )
        mapping[label] = atom_idx
    return mapping


def _rmsd_for_matching_atoms(trc_ref, trc_cmp):
    ref_geom = trc_ref.topology.geometry
    cmp_geom = trc_cmp.topology.geometry
    total_sq = 0.0
    atom_count = 0
    for residue_idx in range(len(trc_cmp.residues.residues)):
        ref_map = _atom_label_map(trc_ref, residue_idx)
        cmp_map = _atom_label_map(trc_cmp, residue_idx)
        missing = set(ref_map) - set(cmp_map)
        assert not missing, f"Missing atoms in residue {residue_idx}: {sorted(missing)}"
        for label, ref_idx in ref_map.items():
            cmp_idx = cmp_map[label]
            dx = ref_geom[ref_idx][0] - cmp_geom[cmp_idx][0]
            dy = ref_geom[ref_idx][1] - cmp_geom[cmp_idx][1]
            dz = ref_geom[ref_idx][2] - cmp_geom[cmp_idx][2]
            total_sq += dx * dx + dy * dy + dz * dz
            atom_count += 1
    if atom_count == 0:
        return 0.0
    return math.sqrt(total_sq / atom_count)


def _residue_rmsd(trc_ref, trc_cmp, residue_idx):
    ref_map = _atom_label_map(trc_ref, residue_idx)
    cmp_map = _atom_label_map(trc_cmp, residue_idx)
    missing = set(ref_map) - set(cmp_map)
    extra = set(cmp_map) - set(ref_map)
    assert not missing, (
        "Non-hydrogen atoms differ in residue "
        f"{residue_idx} (missing={sorted(missing)}, extra={sorted(extra)})."
    )
    if not ref_map:
        return 0.0

    ref_geom = trc_ref.topology.geometry
    cmp_geom = trc_cmp.topology.geometry
    total_sq = 0.0
    for label, ref_idx in ref_map.items():
        cmp_idx = cmp_map[label]
        dx = ref_geom[ref_idx][0] - cmp_geom[cmp_idx][0]
        dy = ref_geom[ref_idx][1] - cmp_geom[cmp_idx][1]
        dz = ref_geom[ref_idx][2] - cmp_geom[cmp_idx][2]
        total_sq += dx * dx + dy * dy + dz * dz
    return math.sqrt(total_sq / len(ref_map))


def _per_residue_rmsd(trc_ref, trc_cmp):
    if not trc_ref.residues.residues:
        return 0.0
    all = []
    for residue_idx in range(len(trc_cmp.residues.residues)):
        all.append(_residue_rmsd(trc_ref, trc_cmp, residue_idx))
    return all


def test_prepare_protein(test_data_dir):
    run_debumped = prepare.protein(
        test_data_dir / "3fln_raw.pdb",
        ph=7.4,
        naming_scheme="AMBER",
        capping_style="truncated",
        truncation_threshold=1,
        debump=True,
        run_opts=RunOpts(
            name="Test prepare-protein 02: Debump",
            tags=["rush-py", "test", "MAPK14"],
        ),
    )
    run_nodebump = prepare.protein(
        test_data_dir / "3fln_raw.pdb",
        ph=7.4,
        naming_scheme="AMBER",
        capping_style="truncated",
        truncation_threshold=1,
        debump=False,
        run_opts=RunOpts(
            name="Test prepare-protein 02: No Debump",
            tags=["rush-py", "test", "MAPK14"],
        ),
    )
    assert_run_collects_and_caches(run_debumped, prepare.ResultRef)
    assert_run_collects_and_caches(run_nodebump, prepare.ResultRef)

    # Load the original PDB into a TRC
    trc_unprepped = _as_trc(from_pdb((test_data_dir / "3fln_raw.pdb").read_text()))

    # Parse into TRC object (single model)
    trc_debumped = run_debumped.fetch()[0]
    trc_nodebump = run_nodebump.fetch()[0]

    rmsd_nodebump = _rmsd_for_matching_atoms(trc_unprepped, trc_nodebump)
    rmsd_debumped = _rmsd_for_matching_atoms(trc_unprepped, trc_debumped)
    assert rmsd_nodebump <= rmsd_debumped, (
        "Expected nodebump RMSD to be lower than debumped RMSD for "
        f"unprepped atoms (nodebump={rmsd_nodebump:.4f}, "
        f"debumped={rmsd_debumped:.4f})."
    )
    residues_rmsd_nodebump = _per_residue_rmsd(trc_unprepped, trc_nodebump)
    residues_rmsd_debumped = _per_residue_rmsd(trc_unprepped, trc_debumped)
    for residue_rmsd_nodebump, residue_rmsd_debumped in zip(
        residues_rmsd_nodebump, residues_rmsd_debumped
    ):
        assert residue_rmsd_nodebump < 1.5, (
            "Expected nodebump residue RMSD to be lower than 1.5 Angstroms: "
            f"debumped={residue_rmsd_debumped:.4f})."
        )
        assert residue_rmsd_debumped < 3.0, (
            "Expected debumped residue RMSD to be lower than 3.0 Angstroms: "
            f"debumped={residue_rmsd_debumped:.4f})."
        )
        assert residue_rmsd_nodebump <= residue_rmsd_debumped, (
            "Expected nodebump residue RMSD to be lower than debumped residue RMSD for "
            f"unprepped residues (nodebump={residue_rmsd_nodebump:.4f}, "
            f"debumped={residue_rmsd_debumped:.4f})."
        )
