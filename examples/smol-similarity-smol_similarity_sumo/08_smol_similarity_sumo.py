"""
Example: smol-similarity nearest-neighbor search

This script demonstrates how to:
1. Submit a smol_similarity_sumo run with JSON object inputs
2. Fetch parsed per-query outputs
3. Print ranked nearest-neighbor SMILES and similarity scores

Tutorial: https://exess.qdx.co/docs/tutorials/08-smol-similarity-smol_similarity_sumo.html

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input files in data/: input_smis.json and smi_partition1.json
"""

from pathlib import Path

from rush import RunOpts, smol_similarity

DATA_DIR = Path(__file__).parent / "data"
PARTITION_FILE = DATA_DIR / "smi_partition1.json"
INPUT_SMILES_FILE = DATA_DIR / "input_smis.json"

print("=" * 60)
print("smol-similarity nearest-neighbor search")
print("=" * 60)

results = smol_similarity.smol_similarity_sumo(
    smol_partitions=[PARTITION_FILE],
    input_smis=INPUT_SMILES_FILE,
    config=smol_similarity.SmolSimilarityConfig(
        min_similarity=0.0,
        min_results=1,
        max_results=10,
    ),
    run_opts=RunOpts(
        name="Tutorial: smol_similarity_sumo",
        tags=["rush-py", "tutorial", "smol-similarity"],
    ),
).fetch()

for query_index, item in enumerate(results):
    print(f"\nQuery {query_index}")
    print("-" * 40)

    if isinstance(item, smol_similarity.ExecutionError):
        print(f"Execution failed at stage {item.stage}: {item.message}")
        continue

    for rank, (smi, score) in enumerate(zip(item.smiles, item.similarities), start=1):
        print(f"{rank:02d}. {smi}  similarity={score:.4f}")

print("\nDone!")
