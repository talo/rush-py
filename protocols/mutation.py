#!/usr/bin/env python3

import csv
import dataclasses
import json
import math
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from time import sleep

import datargs
import httpx
import prettyprinter
from paratus import QDX_AMISS_PATHOGENIC
from prettyprinter import pprint

prettyprinter.install_extras(exclude=["ipython", "django"])

ALIGNED_HUMAN_SEQ = "-------------------MVRMVPVLLSLLLLLGPAVPQENQDGR--YSLTYIYTGLSKHVEDVPAFQALGSLNDLQFFRYNSKDRKSQPMGLWRQVEGMEDWKQDSQLQKAREDIFMETLKDIVEYYNDSNG--------------------SHVLQGRFGCEIENNRSSGAFWKYYYDGKDYIEFNKEIPAWVPFDPAAQITKQKWEAEPVYVQRAKAYLEEECPATLRKYLKYSKNILDRQDPPSVVVTSHQAPGEKKKLKCLAYDFYPGKIDVHWTRAGEVQEPELRGDVLHNGNGTYQSWVVVAVPPQDTAP--YSCHVQHSSLAQPLVVPWEAS----------------"


def aligned_index(seq_a, i_u):
    """
    inputs
    --------
    seq_a: the aligned sequence (with dashes)
    i_u:   the index into the unaligned sequence

    outputs:
    ----
    aligned indices and the amino acid at the index, respectively
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


@dataclass
class Mutation:
    old_amino_acid_name: str
    new_amino_acid_name: str
    chain_id: str
    residue_number: int
    amiss_is_snv: bool | None
    amiss_pathogenicity_score: float | None
    amiss_pathogenicity_class: str | None
    ddmut_ddG_kcal_per_mol: float | None
    ddmut_ddG_kcal_per_mol_reverse: float | None

    @classmethod
    def from_row(cls, row: dict[str, str]):
        return Mutation(
            row["a.a.1"],
            row["a.a.2"],
            "A",
            int(row["position"]),
            row["is_snv"] == "y",
            float(row["pathogenicity score"]),
            row["pathogenicity class"],
            math.nan,
            math.nan,
        )


@dataclass
class Args:
    name: str
    target: str
    region_of_interest: str | None


def main():
    args = datargs.parse(Args)
    n = 0
    mutations = {}  # key = ResidueIndex (int), val: list[AlphaMissenseMutation]
    if args.target in (
        "mAetAle1_1-291"
        "mArtInt1_1-290"
        "mArtLit_1-290"
        "mCenSen1_1-293"
        "mCynHor1_1-291"
        "mLopEvo1_1-293"
        "mPlaHel1_1-293"
        "mStuPar1_1-292"
        "mUroCon1_1-293"
        "pub_GRCm39_1-307"
    ):
        """
        # Paratus variants (only have the sequence)
        with open(Path.cwd() / "paratus_data" / "paratus_seqs.json") as f:
            paratus_seqs = json.load(f)

        def find_seq(seqs, species_name):
            for seq in paratus_seqs["seqs"]:
                if species_name in seq["name"]:
                    return seq
            return None

        seq = find_seq(paratus_seqs["seqs"], args.target.split("_")[0])
        if not seq:
            print("Couldn't find species!")
            exit(1)

        raw_name = "".join(c if (c.isalnum() or c in ("-")) else "_" for c in seq["name"])
        raw_seq = seq["seq"].replace("-", "")
        for msa_residue_number, variants in REGION_OF_INTEREST.items():
            residue_number = unaligned_index(seq["seq"], msa_residue_number - 1)
            aa = seq["seq"][msa_residue_number - 1]
            assert aa == raw_seq[residue_number - 1]  # should match
            print(f"{raw_name},{msa_residue_number},{residue_number},{aa}")
            if residue_number not in mutations:
                mutations[residue_number] = []
            for variant in (v for v in variants if v != aa):
                mutations[residue_number].append(
                    Mutation(
                        aa,
                        variant,
                        "A",
                        residue_number,
                        math.nan,
                        math.nan,
                    )
                )
                n += 1
        """
        pass
    else:
        # UniProt-sourced (full UniProt metadata + AlphaMissense)
        with open(Path.home() / f"Downloads/AlphaMissense/AlphaMissense-Search-{args.target}.tsv") as fd:
            rd = csv.DictReader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                mut = Mutation.from_row(row)
                # if mut.residue_number in TARGET_RESIDUES
                # and (
                #     mut.amiss_pathogenicity_score
                #     and (mut.amiss_pathogenicity_score >= 0.75 or mut.amiss_pathogenicity_score <= 0.25)
                # ):
                this_aligned_index, this_aa = aligned_index(ALIGNED_HUMAN_SEQ, mut.residue_number - 1)
                if mut.residue_number <= 5:
                    pass
                if (
                    this_aligned_index + 1 in QDX_AMISS_PATHOGENIC
                    and mut.new_amino_acid_name in QDX_AMISS_PATHOGENIC[this_aligned_index + 1]
                ):
                    # print(mut.residue_number)
                    # print(mut.old_amino_acid_name, QDX_AMISS_PATHOGENIC[this_aligned_index + 1])
                    # print(mut.old_amino_acid_name, this_aa)
                    assert mut.old_amino_acid_name in QDX_AMISS_PATHOGENIC[this_aligned_index + 1]
                    assert mut.old_amino_acid_name == this_aa
                    n += 1
                    if mut.residue_number not in mutations:
                        mutations[mut.residue_number] = []
                    mutations[mut.residue_number].append(mut)

    # Print mutations to submit
    for ms_for_res in mutations.values():
        for m in ms_for_res:
            print(f"A {m.residue_number:>3} {m.old_amino_acid_name} → {m.new_amino_acid_name}")
    print(f"{n} total mutations to submit.")

    # Write file for mutation data for DDmut submission
    with open(Path.home() / f"{args.target}_mutations_qdx-amiss-pathogenic.txt", "w") as f:
        for ms_for_res in mutations.values():
            for m in ms_for_res:
                f.write(f"A {m.old_amino_acid_name}{m.residue_number}{m.new_amino_acid_name}\n")

    # Make DDMut submission
    # target_path = Path.cwd() / "objects" / "ZAG_species" / f"{args.target}"
    target_path = (
        Path.cwd()
        / "objects"
        / "paratus_druggability_results"
        / "druggability_results"
        / "6_ZAG"
        / "alphafold"
    )
    with (
        open(target_path / f"{args.target}_1.2_alphafold.pdb", "rb") as pdb,
        open(Path.home() / f"{args.target}_mutations_qdx-amiss-pathogenic.txt", "rb") as txt,
    ):
        submission = httpx.post(
            "https://biosig.lab.uq.edu.au/ddmut/api/prediction_list",
            files={
                "pdb_file": pdb,
                "mutations_list": txt,
            },
            data={
                "reverse": "True",
            },
            timeout=60,
        )
        print(f"sent POST to {submission.url}")
        submission_json = submission.json()
        pprint(submission_json)

    # Get DDmut results
    results_url = "https://biosig.lab.uq.edu.au/ddmut/api/prediction_list"
    results_data = {"job_id": submission_json["job_id"]}
    results = httpx.request("GET", results_url, data=results_data)
    print(f"Sent GET to {results.url}")
    results_json = results.json()
    wait_secs = 1
    while "status" in results_json and results_json["status"] == "RUNNING":
        sleep(wait_secs)
        wait_secs *= min(2, 32)
        results = httpx.request("GET", results_url, data=results_data)
        results_json = results.json()
    pprint(results_json)
    with open(Path.home() / f"{args.target}_ddmut_qdx-amiss-pathogenic.json", "w") as f:
        json.dump(results_json, f)

    # Find DDMut prediction for each Mutation object
    with open(Path.home() / f"{args.target}_ddmut_qdx-amiss-pathogenic.json") as f:
        results_json = json.load(f)

    def find_ddmut_prediction(ddmut_results, m) -> tuple[float, float]:
        for ddmut_result in ddmut_results.values():
            mutation_str = f"{m.old_amino_acid_name}{m.residue_number}{m.new_amino_acid_name}"
            if ddmut_result["chain"] == "A" and ddmut_result["mutation"] == mutation_str:
                print(ddmut_result["prediction"], ddmut_result["prediction_reverse"])
                return ddmut_result["prediction"], ddmut_result["prediction_reverse"]
        return math.nan, math.nan

    for ms_for_res in mutations.values():
        for m in ms_for_res:
            m.ddmut_ddG_kcal_per_mol, m.ddmut_ddG_kcal_per_mol_reverse = find_ddmut_prediction(
                results_json, m
            )
            m.old_amino_acid_name = AA_1TO3[m.old_amino_acid_name]
            m.new_amino_acid_name = AA_1TO3[m.new_amino_acid_name]

    # Write final mutation data
    with open(Path.home() / f"{args.name}_{args.target}_mutations_qdx-amiss-pathogenic.json", "w") as f:
        json.dump(
            {
                "protein_name": args.name,
                "uniprot_id": args.target,
                "job_id": submission_json["job_id"],
                "mutations": mutations,
            },
            f,
            indent=2,
            cls=EnhancedJSONEncoder,
        )

    with open(Path.home() / f"{args.name}_{args.target}_mutations_qdx-amiss-pathogenic.csv", "w") as f:
        mutations_list = [m for ms_for_res in mutations.values() for m in ms_for_res]
        writer = csv.writer(f)
        writer.writerow(mutations_list[0].__annotations__.keys())
        for m in mutations_list:
            writer.writerow(m.__dict__.values())


if __name__ == "__main__":
    main()
