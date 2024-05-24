#!/usr/bin/env python3

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import datargs
import httpx

import rush


@dataclass
class CrossRefPDB:
    database: str
    id: str
    method: str
    resolution: str
    chains: str


@dataclass
class UniprotData:
    entry_type: str
    primary_accession: str
    sequence: str
    molecular_weight: int
    pdb_cross_refs: list[CrossRefPDB]


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
    target_id: str
    target_seq_range: set[tuple[int, int] | int]
    # domain_name: str | None  # not always available
    interactors: list[Interactor]


# Parameters

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


def get_uniprot(target: str):
    params = {
        "query": f"organism_name:Human AND reviewed:true AND gene:{target}",
        "fields": "sequence,xref_pdb,xref_intact",  # xref_pfam
        "format": "json",
    }
    response = httpx.get("https://rest.uniprot.org/uniprotkb/stream", params=params)
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
            pdb_cross_refs.append(
                CrossRefPDB(cross_ref_data["database"], cross_ref_data["id"], method, resolution, chains)
            )
        uniprot_data.append(
            UniprotData(
                result["entryType"],
                result["primaryAccession"],
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


def parse_range(x) -> list[tuple[int, int] | int]:
    result = []
    for part in x.split(","):
        if "-" in part:
            a, b = part.split("-")
            a, b = int(a), int(b)
            result.append((a, b))
        else:
            a = int(part)
            result.append(a)
    return result


def get_binding_ixns_from_intact(target_id: str) -> list[BindingInteraction]:
    params = {
        "advancedSearch": "true",
        "interactionTypesFilter": "direct interaction",
        "interactorSpeciesFilter": "Homo sapiens",
        "intraSpeciesFilter": "true",
        "negativeFilter": "POSITIVE_ONLY",
        "query": f"id:{target_id}",
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
                                        range
                                        for sequence_data in feature["sequenceData"]
                                        for range in parse_range(sequence_data["pos"])
                                    ],
                                )
                            )

        for k, interaction_id in interaction_ids.items():
            binding_interactions.append(
                BindingInteraction(
                    interaction_id,
                    target_id,
                    target_seq_range=set(
                        sequence_range
                        for participant in interaction_participants[k]
                        if participant.id == f"uniprotkb_{target_id}"
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


def get_binding_ixns_from_pdbe(cross_refs: list[str]):
    pass


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


@dataclass
class Args:
    target: str


def main():
    args = datargs.parse(Args)

    #### Starting with UniProt seqs

    # 1.0: Obtain input (FASTA)
    uniprot_data = get_uniprot(args.target)
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
    (folded_conformers_handle,) = client.colabfold_fold(
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
    with open(WORK_DIR / "objects" / f"{uniprot_id}_2.0_pocket_data.json", "w") as f:
        json.dump(p2rank_prediction_handle.get(), f, indent=2)
    pymol_viz_handle.download(
        filename=f"{uniprot_id}_2.0_pocket_viz.tar.gz",
        overwrite=True,
    )

    #### Starting with cross-refs (i.e. PDB files)

    for pdb_cross_ref in uniprot_data.pdb_cross_refs:
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

        # 2.0: Evaluate pockets
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

    #### Binding site analysis

    # 3.1: get IntAct data (mostly Protein - Protein, Protein - DNA/RNA, etc.)
    _intact_binding_ixns = get_binding_ixns_from_intact(uniprot_id)

    # 3.2: get PDB binding site data (mostly Protein - Ligand/SMOL)
    _pdb_binding_ixns = get_binding_ixns_from_pdbe(uniprot_data.pdb_cross_refs)


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
