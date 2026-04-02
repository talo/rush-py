from pathlib import Path

from rush import RunOpts, smol_similarity
from tests._module_test_utils import assert_run_collects_and_caches


def test_smol_similarity_sumo(test_data_dir: Path):
    module_data = test_data_dir / "smol_similarity"

    run = smol_similarity.smol_similarity_sumo(
        smol_partitions=[module_data / "smi_partition1.json"],
        input_smis=module_data / "input_smis.json",
        config=smol_similarity.SmolSimilarityConfig(
            min_similarity=0.0,
            min_results=1,
            max_results=10,
        ),
        run_opts=RunOpts(
            name="Rush-Py Test smol_similarity_sumo 01",
            tags=["rush-py", "test", "smol-similarity"],
        ),
    )

    ref = assert_run_collects_and_caches(run, smol_similarity.ResultRef)
    assert len(ref) == 1

    fetched = run.fetch()
    assert len(fetched) == 1
    assert isinstance(fetched[0], smol_similarity.Result)
    assert len(fetched[0].smiles) == len(fetched[0].similarities)
    assert len(fetched[0].smiles) > 0
    assert all(0.0 <= score <= 1.0 for score in fetched[0].similarities)

    saved = run.save()
    assert len(saved) == 1
    assert isinstance(saved[0], smol_similarity.ResultPaths)
    assert saved[0].output.exists()
