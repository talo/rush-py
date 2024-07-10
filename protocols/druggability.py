#!/usr/bin/env python3

import dataclasses
import gzip
import json
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Literal

import datargs
import httpx
import prettyprinter

import rush

from blosum import BLOSUM62
from paratus import QDX_AMISS_PATHOGENIC
# from prettyprinter import pprint

prettyprinter.install_extras(exclude=["ipython", "django"])


def aligned_index(seq_a, i_u):
    """
    seq_a is the aligned sequence (with dashes)
    i_u is the index into the unaligned sequence so 113 / 129 / 157 for our cases since the index starts at 0
    output (i_a, seq_a[i_a]) are the aligned indices and the amino acid at the index, respectively
    """
    seq_u = seq_a.replace("-", "")
    walk_a = 0
    walk_u = 0
    while walk_u != i_u or seq_a[walk_a] != seq_u[i_u]:
        if seq_a[walk_a] == "-":
            walk_a += 1
        elif seq_a[walk_a] == seq_u[walk_u]:
            walk_a += 1
            walk_u += 1
    i_a = walk_a
    assert i_u == walk_u
    assert i_u <= i_a
    assert seq_u[i_u] == seq_a[i_a]
    return (i_a, seq_a[i_a])


def unaligned_index(seq_a, i_a):
    return len(seq_a[: i_a + 1].replace("-", ""))


AA_1TO3 = {
    "C": "CYS",
    "D": "ASP",
    "S": "SER",
    "Q": "GLN",
    "K": "LYS",
    "I": "ILE",
    "P": "PRO",
    "T": "THR",
    "F": "PHE",
    "N": "ASN",
    "G": "GLY",
    "H": "HIS",
    "L": "LEU",
    "R": "ARG",
    "W": "TRP",
    "A": "ALA",
    "V": "VAL",
    "E": "GLU",
    "Y": "TYR",
    "M": "MET",
}


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return dataclasses.asdict(o)
        elif isinstance(o, set):
            return sorted(list(o))
        return super().default(o)


#### These are for UniProt information cross refs, etc. about the target


@dataclass
class CrossRefPDB:
    id: str
    method: str
    resolution: str
    chains: str


@dataclass
class UniprotData:
    entry_type: str
    primary_accession: str
    sequence: str
    aligned_sequence: str
    molecular_weight: int
    pdb_cross_refs: list[CrossRefPDB]


#### These are for PDBe binding interaction data


@dataclass
class IntActInteractorData:
    type: Literal["Protein", "DNA", "RNA", "Ligand", "HOH", "Ion"]
    identifier_db: str
    identifier_id: str


@dataclass
class IntActInteractionData:
    id: str  # by interactorRef
    category: str
    sequence_ranges: list[tuple[int, int] | int]


@dataclass
class Interactor:
    type: Literal["Protein", "DNA", "RNA", "Ligand", "HOH", "Ion"]
    identifier_db: str
    identifier_id: str
    category: str
    sequence_ranges: list[tuple[int, int] | int]


@dataclass
class BindingInteraction:
    interaction_id: str
    uniprot_id: str
    target_seq_range: set[tuple[int, int] | int]
    # domain_name: str | None  # not always available
    interactors: list[Interactor]


#### These are for PDBe binding interaction data


@dataclass
class PDBeBindingSiteAminoAcid:
    amino_acid_name: str
    chain_id: str
    residue_number: int
    author_chain_id: int
    author_residue_number: int
    author_insertion_code: int


@dataclass
class PDBeBindingSiteLigand:
    residue_name: str
    chain_id: str
    residue_number: int
    author_residue_number: int


@dataclass
class BindingSite:
    pdb_id: str
    site_id: str
    target_seq_range: set[tuple[int, int] | int]
    target_residues: list[PDBeBindingSiteAminoAcid]
    ligands: list[PDBeBindingSiteLigand]
    details: str


def get_uniprot(target: str):
    params = {
        "query": f"organism_name:Human AND reviewed:true AND gene:{target}",
        "fields": "sequence,xref_pdb,xref_intact",  # xref_pfam
        "format": "json",
    }
    response = httpx.get("https://rest.uniprot.org/uniprotkb/stream", params=params)
    print(response.url)
    response_json = response.json()
    uniprot_data = []
    for result in response_json["results"]:
        pdb_cross_refs = []
        for cross_ref_data in result["uniProtKBCrossReferences"]:
            if cross_ref_data["database"] != "PDB":
                continue
            method = ""
            resolution = ""
            chains = ""
            for property in cross_ref_data["properties"]:
                if property["key"] == "Method":
                    method += property["value"]
                if property["key"] == "Resolution":
                    resolution += property["value"]
                if property["key"] == "Chains":
                    chains += property["value"]
            pdb_cross_refs.append(CrossRefPDB(cross_ref_data["id"], method, resolution, chains))
        uniprot_data.append(
            UniprotData(
                result["entryType"],
                result["primaryAccession"],
                result["sequence"]["value"],
                result["sequence"]["value"],
                result["sequence"]["molWeight"],
                pdb_cross_refs,
            )
        )
    assert len(uniprot_data) == 1
    return uniprot_data[0]


def get_pdb(id: str):
    print(f"https://files.rcsb.org/download/{id}.pdb.gz")
    response = httpx.get(f"https://files.rcsb.org/download/{id}.pdb.gz", timeout=30.0)
    return str(gzip.decompress(response.content), "utf-8")


def parse_range(idx_range) -> list[tuple[int, int] | int]:
    result = []
    if isinstance(idx_range, int):
        return [idx_range]
    for part in idx_range.split(","):
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            result.append((a, b))
        else:
            a = int(part)
            result.append(a)
    return result


def get_binding_ixns_from_intact(uniprot_id: str) -> list[BindingInteraction]:
    params = {
        "advancedSearch": "true",
        "interactionTypesFilter": "direct interaction",
        "interactorSpeciesFilter": "Homo sapiens",
        "intraSpeciesFilter": "true",
        "negativeFilter": "POSITIVE_ONLY",
        "query": f"id:{uniprot_id}",
    }
    response = httpx.post(
        "https://www.ebi.ac.uk/intact/ws/interaction/findInteractionWithFacet", params=params
    )
    print(response.url)
    response_json = response.json()
    binding_interactions = []
    for result in response_json["data"]["content"]:
        response = httpx.get(
            f"https://www.ebi.ac.uk/intact/ws/graph/export/interaction/{result['ac']}",
            params={"format": "miJSON"},
        )
        print(response.url)
        response_json = response.json()
        interactors = {}
        interaction_ids = {}
        interaction_participants = {}
        for obj in response_json["data"]:
            # handle interactors
            if obj["object"] == "interactor":
                interactors[obj["id"]] = IntActInteractorData(
                    obj["type"]["name"],
                    obj["identifier"]["db"],
                    obj["identifier"]["id"],
                )

            # handle interations
            if obj["object"] == "interaction":
                for participant in obj["participants"]:
                    if "features" in participant:
                        for feature in participant["features"]:
                            if feature["category"] != "bindingSites":
                                # skip non binding site features ("experimentalFeatures, ptms")
                                continue

                            # get a unique key for this logical interaction
                            if "linkedFeatures" in feature:
                                k = frozenset([feature["id"]] + feature["linkedFeatures"])
                            else:
                                k = frozenset([feature["id"]])

                            if k not in interaction_ids:
                                interaction_ids[k] = obj["id"]
                                interaction_participants[k] = []
                            interaction_participants[k].append(
                                IntActInteractionData(
                                    participant["interactorRef"],
                                    feature["category"],
                                    [
                                        sequence_range
                                        for sequence_data in feature["sequenceData"]
                                        for sequence_range in parse_range(sequence_data["pos"])
                                    ],
                                )
                            )

        for k, interaction_id in interaction_ids.items():
            binding_interactions.append(
                BindingInteraction(
                    interaction_id,
                    uniprot_id,
                    target_seq_range=set(
                        sequence_range
                        for participant in interaction_participants[k]
                        if participant.id == f"uniprotkb_{uniprot_id}"
                        for sequence_range in participant.sequence_ranges
                    ),
                    interactors=[
                        Interactor(
                            interactors[participant.id].type,
                            interactors[participant.id].identifier_db,
                            interactors[participant.id].identifier_id,
                            participant.category,
                            participant.sequence_ranges,
                        )
                        for participant in interaction_participants[k]
                    ],
                )
            )

    return binding_interactions


@dataclass
class BindingPocket:
    target_seq_range: set[tuple[int, int] | int]
    center: tuple[float, float, float]
    probability: float


def get_binding_regions_from_p2rank(uniprot_id: str, probability_threshold: float) -> list[BindingPocket]:
    with open(WORK_DIR / "objects" / f"{uniprot_id}_2.0_pocket_data.json") as f:
        p2rank_data = json.load(f)

    return [
        BindingPocket(
            {int(residue_id.split("_")[1]) for residue_id in pocket["residue_ids"]},
            (pocket["center_x"], pocket["center_y"], pocket["center_z"]),
            pocket["probability"],
        )
        for pocket in p2rank_data["pockets"]
        if pocket["probability"] >= probability_threshold
    ]


def get_binding_sites_from_pdbe(pdb_cross_refs: list[str]) -> list[BindingSite]:
    binding_sites = []
    # for pdb_cross_ref_id in ["3RN2", "3RN5", "3VD8", "4O7Q", "6MB2", "7K3R"]:
    for pdb_cross_ref in pdb_cross_refs:
        pdb_cross_ref_id = pdb_cross_ref.id
        response = httpx.get(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/binding_sites/{pdb_cross_ref_id}")
        print(response.url)
        response_json = response.json()
        for pdb_id, binding_sites_data in response_json.items():
            for binding_site_data in binding_sites_data:
                binding_site_aas = []
                for site_residue in binding_site_data["site_residues"]:
                    # TODO: check the entity ID against the assembly, that it's a "polypeptide(L)"?
                    if site_residue["entity_id"] == 1:
                        binding_site_aas.append(
                            PDBeBindingSiteAminoAcid(
                                site_residue["chem_comp_id"],
                                site_residue["chain_id"],
                                site_residue["residue_number"],
                                site_residue["struct_asym_id"],
                                site_residue["author_residue_number"],
                                site_residue["author_insertion_code"],
                            )
                        )
                binding_site_ligs = []
                for ligand_residue in binding_site_data["ligand_residues"]:
                    # TODO: Check this against ligand REST query or assembly, that it's a "bound"?
                    binding_site_ligs.append(
                        PDBeBindingSiteLigand(
                            ligand_residue["chem_comp_id"],
                            ligand_residue["chain_id"],
                            ligand_residue["residue_number"],
                            ligand_residue["author_residue_number"],
                        )
                    )
                binding_sites.append(
                    BindingSite(
                        pdb_id,
                        binding_site_data["site_id"],
                        set(
                            sequence_range
                            for binding_site_aa in binding_site_aas
                            for sequence_range in parse_range(binding_site_aa.residue_number)
                        ),
                        binding_site_aas,
                        binding_site_ligs,
                        binding_site_data["details"],
                    )
                )

    return binding_sites


def compute_percent_overlapping(reference_range, query_range) -> float:
    def expand_range(idx_range) -> list[float]:
        expansion = []
        for r in idx_range:
            if isinstance(r, tuple):
                expansion.extend(range(r[0], r[1] + 1))
            elif isinstance(r, int):
                expansion.append(r)
        return expansion

    reference_idxs = expand_range(reference_range)
    query_idxs = expand_range(query_range)

    if len(reference_idxs) == 0:
        return 0.0

    return sum(1 if query_idx in reference_idxs else 0 for query_idx in query_idxs) / len(reference_idxs)


def compute_scores(
    region_of_interest, intact_binding_ixns, p2rank_binding_pockets, pdbe_binding_sites
) -> tuple[float, float, float, float]:
    # ligand <-> region of interest
    lig_roi_score = 0.0
    for binding_site in pdbe_binding_sites:
        reference_seq_range = region_of_interest
        query_seq_range = binding_site.target_seq_range
        pct_overlap = compute_percent_overlapping(reference_seq_range, query_seq_range)
        if pct_overlap > 0.0:
            lig_roi_score += 1 + pct_overlap

    # ligand <-> binding interaction regions (experimental)
    lig_bixn_score = 0.0
    for binding_site in pdbe_binding_sites:
        for binding_ixn in intact_binding_ixns:
            reference_seq_range = binding_ixn.target_seq_range
            query_seq_range = binding_site.target_seq_range
            pct_overlap = compute_percent_overlapping(reference_seq_range, query_seq_range)
            if pct_overlap > 0.0:
                lig_bixn_score += 1 + pct_overlap

    # ligand <-> binding pockets (predicted)
    lig_bpocket_score = 0.0
    for binding_site in pdbe_binding_sites:
        for binding_region in p2rank_binding_pockets:
            reference_seq_range = binding_region.target_seq_range
            query_seq_range = binding_site.target_seq_range
            pct_overlap = compute_percent_overlapping(reference_seq_range, query_seq_range)
            if pct_overlap > 0.0:
                lig_bpocket_score += 1 + pct_overlap

    # ligand <-> other ligands
    lig_lig_score = 0.0
    # used = {}
    for i in range(len(pdbe_binding_sites)):
        for j in range(i, len(pdbe_binding_sites)):
            seq_range_i = pdbe_binding_sites[i].target_seq_range
            seq_range_j = pdbe_binding_sites[j].target_seq_range
            avg_pct_overlap = (
                compute_percent_overlapping(seq_range_i, seq_range_j)
                + compute_percent_overlapping(seq_range_i, seq_range_j)
            ) / 2
            if avg_pct_overlap > 0.0:
                lig_lig_score += 1 + avg_pct_overlap
                # Use each identical pair only once
                # k = tuple((
                #     pdbe_binding_sites[i].pdb_id,
                #     frozenset(ligand.residue_name for ligand in pdbe_binding_sites[i].ligands),
                #     pdbe_binding_sites[j].pdb_id,
                #     frozenset(ligand.residue_name for ligand in pdbe_binding_sites[j].ligands),
                # ))
                # if k not in used:
                #     used[k] = avg_pct_overlap
                #     lig_lig_score += 1 + avg_pct_overlap
                # else:
                #     print(f"reusing {k}!")

    return (lig_roi_score, lig_bixn_score, lig_bpocket_score, lig_lig_score)


def read_fasta(filepath: Path) -> dict[str, str]:
    """Read a FASTA file. See:
    https://blast.ncbi.nlm.nih.gov/doc/blast-topics/queryinpanddatasel.html#accepted-input-formats
    """
    fasta_sequences = {}
    current_key = ""
    current_seq: list[str] = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                if len(current_seq) != 0 and current_key not in fasta_sequences:
                    fasta_sequences[current_key] = "".join(current_seq)
                current_key = line[1:].rstrip()
                current_seq = []
            elif line.startswith(";"):
                # ignore comments, if any are present
                continue
            else:
                segments = line.split()
                # the first space-separated segment can be numeric
                if segments[0].isdigit():
                    segments = segments[0]
                # the rest need to be a valid FASTA nucleic acid or amino acid code
                subseq = "".join(segments)
                if set(subseq).issubset("ABCDEFGHIKLMNPQRSTUVWYZX*-"):
                    current_seq.append(subseq)
                else:
                    pass

        if len(current_seq) != 0 and current_key not in fasta_sequences:
            fasta_sequences[current_key] = "".join(current_seq)

        return fasta_sequences


def fasta_from_pdb(pdb_str):
    from Bio import PDB
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    structure = pdb_str
    models = []
    for model in structure:
        chains = []
        for chain in model:
            sequence = ""
            for residue in chain:
                if PDB.is_aa(residue):
                    sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
            seq_record = SeqRecord(Seq(sequence), id=chain.id)
            chains.append(seq_record.format("fasta"))
        models.append(chains)

    return models


#### Parameters

WORK_DIR = Path.cwd()

COLABFOLD_SEARCH_RUN_CONFIG = {
    "resources": {
        "gpus": 0,
        "cpus": 48,
        "mem": 128 * 1024,
        "storage_mounts": ["gdata/if89"],
        "walltime": 360,
    },
    "target": "GADI",
}

COLABFOLD_FOLD_RUN_CONFIG = {
    "resources": {
        "gpus": 1,
        "storage_mounts": ["gdata/if89"],
        "walltime": 360,
    },
    "target": "GADI",
}

PREPARE_PROTEIN_RUN_CONFIG = {}

P2RANK_RUN_CONFIG = {}


@dataclass
class Args:
    target: str
    region_of_interest: str | None


def main():
    args = datargs.parse(Args)

    #### Starting with UniProt seqs

    if args.target == "paratus_seqs":
        with open(Path.cwd() / "paratus_data" / "paratus_seqs.json") as f:
            paratus_seqs = json.load(f)
        uniprot_data = {}
        for seq in paratus_seqs["seqs"]:
            raw_name = "".join(c if (c.isalnum() or c in ("-")) else "_" for c in seq["name"])
            raw_seq = seq["seq"].replace("-", "")
            # if (
            #     raw_seq
            #     == "MVRMVPVLLSLLLLLGPAVPQENQDGRYSLTYIYTGLSKHVEDVPAFQALGSLNDLQFFRYNSKDRKSQPMGLWRQVEGMEDWKQDSQLQKAREDIFMETLKDIVEYYNDSNGSHVLQGRFGCEIENNRSSGAFWKYYYDGKDYIEFNKEIPAWVPFDPAAQITKQKWEAEPVYVQRAKAYLEEECPATLRKYLKYSKNILDRQDPPSVVVTSHQAPGEKKKLKCLAYDFYPGKIDVHWTRAGEVQEPELRGDVLHNGNGTYQSWVVVAVPPQDTAPYSCHVQHSSLAQPLVVPWEAS"
            # ):
            #     print(aligned_index(seq["seq"], 157))
            for residue_num in QDX_AMISS_PATHOGENIC.keys():
                print(
                    f"{raw_name},{unaligned_index(seq["seq"], residue_num - 1)},{seq["seq"][residue_num - 1]}"
                )
            # print(raw_name)
            # print(raw_seq)
            uniprot_data[raw_name] = UniprotData("", raw_name, raw_seq, seq["seq"], 0, [])
        # uniprot_data = uniprot_data[0]
    else:
        # 1.0: Obtain input (FASTA)
        uniprot_data = get_uniprot(args.target)

    print(len(uniprot_data))
    # mAetAle1_1-291
    # mCynHor1_1-291
    # mArtInt1_1-290
    # mArtLit_1-290
    # mCenSen1_1-293
    # mPlaHel1_1-293
    # mStuPar1_1-292
    # mUroCon1_1-293
    # mLopEvo1_1-293
    uniprot_data = uniprot_data["pub_GRCh38_1-298"]

    for residue_number, mutations in QDX_AMISS_PATHOGENIC.items():
        for m in mutations:
            w = uniprot_data.aligned_sequence[residue_number - 1]
            if m != w:
                print(f"{residue_number},{w},{m},{BLOSUM62[w][m]}")

    uniprot_id = uniprot_data.primary_accession
    print(f"{uniprot_id=}")
    print(f"{uniprot_data.sequence=}")

    client = rush.build_blocking_provider_with_functions(
        batch_tags=["paratus", "protocol", "druggability", uniprot_data.primary_accession],
        workspace=WORK_DIR,
        restore_by_default=True,
    )

    # 1.1: MSA
    (msa_handle,) = client.mmseqs2(
        {"fasta": [uniprot_data.sequence]},
        **COLABFOLD_SEARCH_RUN_CONFIG,
    )

    # 1.2: Produce 3D structure
    (folded_conformers_handle, folded_pdbs_handle) = client.colabfold_fold(
        msa_handle,
        **COLABFOLD_FOLD_RUN_CONFIG,
    )
    (folded_conformer_handle,) = client.pick_conformer(folded_conformers_handle, 0)

    # 2.0: Evaluate pockets
    (p2rank_prediction_handle, pymol_viz_handle) = client.p2rank(
        folded_conformer_handle,
        True,
        **P2RANK_RUN_CONFIG,
    )

    # 2.1: Download everything & show evaluation
    msa_handle.download(
        filename=f"{uniprot_id}_1.1_msa.tar.gz",
        overwrite=True,
    )
    folded_conformer_handle.download(
        filename=f"{uniprot_id}_1.2_conformer.json",
        overwrite=True,
    )
    folded_pdbs_handle.download(
        filename=f"{uniprot_id}_alphafold.tar.gz",
        overwrite=True,
    )
    with open(WORK_DIR / "objects" / f"{uniprot_id}_2.0_pocket_data.json", "w") as f:
        json.dump(p2rank_prediction_handle.get(), f, indent=2)
    pymol_viz_handle.download(
        filename=f"{uniprot_id}_2.0_pocket_viz.tar.gz",
        overwrite=True,
    )

    """
    #### Starting with cross-refs (i.e. PDB files)

    for pdb_cross_ref in uniprot_data.pdb_cross_refs:
        if pdb_cross_ref.id == "3DLS":
            continue
        # 1.2: Get 3D structure from cross-referenced data
        pdb_path = WORK_DIR / "objects" / f"{uniprot_id}-{pdb_cross_ref.id}_1.2_structure.pdb"
        if not pdb_path.is_file():
            pdb_path.write_text(get_pdb(pdb_cross_ref.id))

        # 1.3: Clean up structure
        (prepared_structures_handle, _) = client.prepare_protein(
            pdb_path,
            None,
            None,
            tags=[pdb_cross_ref.id],
            **PREPARE_PROTEIN_RUN_CONFIG,
        )
        (prepared_structure_handle,) = client.pick_conformer(
            prepared_structures_handle, 0, tags=[pdb_cross_ref.id]
        )

        # # 2.0: Evaluate pockets
        (p2rank_prediction_handle, pymol_viz_handle) = client.p2rank(
            prepared_structure_handle,
            True,
            tags=[pdb_cross_ref.id],
            **P2RANK_RUN_CONFIG,
        )

        # 2.1: Download everything & show evaluation
        prepared_structure_handle.download(
            filename=f"{uniprot_id}-{pdb_cross_ref.id}_1.3_prepared.json",
            overwrite=True,
        )
        with open(WORK_DIR / "objects" / f"{uniprot_id}-{pdb_cross_ref.id}_2.0_pocket_data.json", "w") as f:
            json.dump(p2rank_prediction_handle.get(), f, indent=2)
        pymol_viz_handle.download(
            filename=f"{uniprot_id}-{pdb_cross_ref.id}_2.0_pocket_viz.tar.gz",
            overwrite=True,
        )

    """

    #### Binding site analysis

    """
    region_of_interest = set((0, len(uniprot_data.sequence)))
    if args.region_of_interest:
        region_of_interest = parse_range(args.region_of_interest)

    # 3.0a: Get IntAct data (mostly Protein - Protein, Protein - DNA/RNA, etc.)
    intact_binding_ixns = get_binding_ixns_from_intact(uniprot_id)
    with open(WORK_DIR / "objects" / f"{uniprot_id}_3.0a_intact_binding_ixns.json", "w") as f:
        json.dump(intact_binding_ixns, f, indent=2, cls=EnhancedJSONEncoder)
    """
    # 3.0b: Parse P2Rank data (predicted potential binding pockets)
    p2rank_pockets = get_binding_regions_from_p2rank(uniprot_id, probability_threshold=0.25)
    with open(WORK_DIR / "objects" / f"{uniprot_id}_3.0b_p2rank_pockets.json", "w") as f:
        json.dump(p2rank_pockets, f, indent=2, cls=EnhancedJSONEncoder)
    """
    # 3.0c: Get PDB binding site data (mostly Protein - Ligand/SMOL)
    pdbe_binding_sites = get_binding_sites_from_pdbe(uniprot_data.pdb_cross_refs)
    with open(WORK_DIR / "objects" / f"{uniprot_id}_3.0c_pdbe_binding_sites.json", "w") as f:
        json.dump(pdbe_binding_sites, f, indent=2, cls=EnhancedJSONEncoder)

    # 3.1: Compute scores
    roi_score, bixn_score, bpocket_score, bsite_score = compute_scores(
        region_of_interest,
        intact_binding_ixns,
        p2rank_pockets,
        pdbe_binding_sites,
    )
    score_dict = {
        "region_of_interest_score": round(roi_score, 2),
        "functional_domain_score": round(bixn_score, 2),
        "binding_pocket_score": round(bpocket_score, 2),
        "ligandability_score": round(bsite_score, 2),
    }
    print(json.dumps(score_dict, indent=2))
    with open(WORK_DIR / "objects" / f"{uniprot_id}_3.1_druggability_score.json", "w") as f:
        json.dump(
            score_dict,
            f,
            indent=2,
            cls=EnhancedJSONEncoder,
        )
    """


if __name__ == "__main__":
    main()


# Next steps:
#   x Uniprot + RCSB, don't filter the RCSB stuff
#   - rank cross-refs based on quality; matching against sequence subsections, number of ligands, etc.
#   - does p2rank give real binding sites,
#       - identifies places part of binding interfaces p-p or p-l interaction?
#   - run PLIP whenever ligands are present
#   - frequently-occuring critical residues
#   - "coherence" or "consistency" of binding sites; is the same place consistently found?
#   - # binding sites
#   - # binding sites part of relevant pathway
#   - # of recurring critical residues
#   - # check for occlusion by P-P interfaces or other unrelated binding
#
# Next:
#   - Mutation simulations; wild-type vs mutant and doing a comparison
#
# Final output:
#   - Rank or score of targets. No need for fancy viz or anything of specific tool outs.
#   - Walter output: executive summary as if done by a comp chemist.
