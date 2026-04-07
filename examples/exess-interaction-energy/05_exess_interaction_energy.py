"""
Example: EXESS Interaction Energy

This script demonstrates how to:
1. Compute fragment-based interaction energy between a ligand and its environment
2. Prepare a complex from PDB and run an end-to-end interaction energy calculation

Tutorial: https://exess.qdx.co/docs/tutorials/05-exess-interaction-energy.html

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input file: tyk2_ejm_31_t.json (provided in data/)
"""

from pathlib import Path

from rush import RunOpts, exess, get_fragments_near_fragment, prepare

# ===== Example 1: Fragment-based interaction energy =====
print("=" * 60)
print("Example 1: Fragment-based interaction energy")
print("=" * 60)

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "interaction-energy-outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ⚠️ TUTORIAL ONLY: STO-3G is a minimal basis set used here for speed/demonstration.
# It is NOT suitable for research or production use. For real work, use at least
# cc-pVDZ or larger (e.g., cc-pVTZ, aug-cc-pVDZ) with an appropriate method.

run = exess.interaction_energy(
    DATA_DIR / "tyk2_ejm_31_t.json",
    93,  # This is the index of the fragment that contains the ligand
    method="RestrictedHF",
    basis="STO-3G",
    frag_keywords=exess.FragKeywords(
        level="Trimer",
        dimer_cutoff=10.0,
        trimer_cutoff=5.0,
        cutoff_type="Centroid",
        distance_metric="Min",
    ),
    run_opts=RunOpts(
        name="Tutorial: Interaction Energy Basic",
    ),
)

# Extract and display results
result = run.fetch()
print(f"Interaction energy: {result.calc.qmmbe.expanded_hf_energy}")


# ===== Example 2: End-to-end from PDB =====
print()
print("=" * 60)
print("Example 2: End-to-end interaction energy from PDB")
print("=" * 60)


# Step 1: Prepare the system
trc_ref = prepare.protein_ligand(
    DATA_DIR / "1hsg.pdb",
    ligand_names=["MK1", "HOH"],
    debump=None,
    run_opts=RunOpts(name="Tutorial: Interaction Energy E2E - Prepare Complex"),
).collect()[0]
trc = trc_ref.fetch()

# Print the charged amino acids
print("Charged amino acids:")
for i, (res_name, formal_charge) in enumerate(
    zip(
        trc.residues.seqs,
        trc.topology.fragment_formal_charges or [0 for _ in trc.residues.seqs],
    )
):
    if int(formal_charge) != 0:
        print(f"{i:>4} {res_name}: {int(formal_charge):+}")

# Step 2: Find ligand fragment index + nearby pocket
lig_idx = trc.residues.seqs.index("MK1")
frag_idcs = get_fragments_near_fragment(trc.topology, lig_idx, 5.0) + [lig_idx]

# Step 3: Run interaction energy

# ⚠️ TUTORIAL ONLY: STO-3G is a minimal basis set used here for speed/demonstration.
# It is NOT suitable for research or production use. For real work, use at least
# cc-pVDZ or larger (e.g., cc-pVTZ, aug-cc-pVDZ) with an appropriate method.

run = exess.interaction_energy(
    trc_ref,
    lig_idx,
    method="RestrictedHF",
    basis="STO-3G",
    frag_keywords=exess.FragKeywords(
        level="Dimer",
        included_fragments=frag_idcs,
    ),
    run_opts=RunOpts(name="Tutorial: Interaction Energy E2E - EXESS"),
)

# Extract and display results
res = run.fetch()
print(f"Interaction energy: {res.calc.qmmbe.expanded_hf_energy}")
