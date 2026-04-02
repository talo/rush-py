"""
smol-similarity module for the Rush Python client.

This module runs nearest-neighbor SMILES similarity search for one or more query
SMILES against one or more partition objects.

Usage::

    from rush import smol_similarity

    run = smol_similarity.smol_similarity_sumo(
        smol_partitions=["partition_a.json", "partition_b.json"],
        input_smis="queries.json",
    )
    result = run.fetch()
"""

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Literal, Self

from gql.transport.exceptions import TransportQueryError

from ._rex import optional_str
from .objects import RushObject, upload_object
from .runs import Run, RunOpts, RunSpec
from .session import _submit_rex


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmolSimilarityConfig:
    """Optional runtime configuration for smol-similarity.

    Attributes:
        min_similarity: Optional lower bound for tanimoto similarity in [0, 1].
        min_results: Minimum number of rows returned per query.
        max_results: Maximum number of rows returned per query.
    """

    min_similarity: float | None = None
    min_results: int | None = None
    max_results: int | None = None


# ---------------------------------------------------------------------------
# Result/error types
# ---------------------------------------------------------------------------



@dataclass(frozen=True)
class ExecutionError:
    """Per-query execution failure for a single input SMILES."""

    stage: Literal["Subprocess", "OutputParse"]
    message: str

    @classmethod
    def from_raw(cls, raw: Any) -> Self:
        if not isinstance(raw, dict):
            raise ValueError(
                "smol_similarity_sumo item Err must be a dict, "
                f"got {type(raw).__name__}"
            )

        stage = raw.get("stage")
        message = raw.get("message")
        if stage not in ("Subprocess", "OutputParse") or not isinstance(message, str):
            raise ValueError(
                "smol_similarity_sumo item Err missing valid stage/message fields"
            )

        return cls(stage=stage, message=message)


@dataclass(frozen=True)
class Result:
    """Parsed similarity output for one query SMILES."""

    smiles: list[str]
    similarities: list[float]


@dataclass(frozen=True)
class ResultPaths:
    """Workspace path for a saved per-query result object."""

    output: Path


@dataclass(frozen=True)
class ItemResultRef:
    """Reference to one per-query result, which may be success or error."""

    output: RushObject | None
    error: ExecutionError | None

    @classmethod
    def from_raw_output(cls, raw: Any) -> Self:
        if not isinstance(raw, dict) or len(raw) != 1:
            raise ValueError(
                "smol_similarity_sumo item output should be Result{Ok|Err}, "
                f"got {type(raw).__name__}"
            )

        if "Ok" in raw:
            value = raw["Ok"]
            if not isinstance(value, dict):
                raise ValueError(
                    "smol_similarity_sumo item Ok must be an object descriptor dict"
                )
            return cls(output=RushObject.from_dict(value), error=None)

        if "Err" in raw:
            return cls(output=None, error=ExecutionError.from_raw(raw["Err"]))

        raise ValueError("smol_similarity_sumo item output must have Ok or Err")

    def fetch(self) -> Result | ExecutionError:
        if self.error is not None:
            return self.error

        if self.output is None:
            raise ValueError("smol_similarity_sumo item missing both output and error")

        payload = self.output.fetch_list()
        if len(payload) != 2:
            raise ValueError(
                "smol_similarity_sumo item payload must be [smiles, similarities], "
                f"got list with {len(payload)} items"
            )

        raw_smiles, raw_similarities = payload
        if not isinstance(raw_smiles, list) or not isinstance(raw_similarities, list):
            raise ValueError(
                "smol_similarity_sumo item payload entries must both be lists"
            )

        smiles = [str(smi) for smi in raw_smiles]
        similarities = [float(similarity) for similarity in raw_similarities]
        return Result(smiles=smiles, similarities=similarities)

    def save(self) -> ResultPaths | ExecutionError:
        if self.error is not None:
            return self.error

        if self.output is None:
            raise ValueError("smol_similarity_sumo item missing both output and error")

        return ResultPaths(output=self.output.save())


@dataclass(frozen=True)
class ResultRef:
    """Reference to per-query smol-similarity outputs."""

    items: list[ItemResultRef]

    def __getitem__(self, index: int) -> ItemResultRef:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[ItemResultRef]:
        return iter(self.items)

    @classmethod
    def from_raw_output(cls, res: Any) -> Self:
        """Parse raw ``collect_run`` output into a ``ResultRef``."""
        if not isinstance(res, list) or len(res) == 0:
            raise ValueError(
                "smol_similarity_sumo should return a non-empty list of per-query results, "
                f"got {type(res).__name__}"
                f" with {len(res) if isinstance(res, list) else '?'} items"
            )

        return cls(items=[ItemResultRef.from_raw_output(item) for item in res])


    def fetch(self) -> list[Result | ExecutionError]:
        """Download outputs and parse per-query results/errors."""
        return [item.fetch() for item in self.items]

    def save(self) -> list[ResultPaths | ExecutionError]:
        """Download outputs and save per-query result objects to the workspace."""
        return [item.save() for item in self.items]


def _to_uploaded_json_object(value: Path | str | RushObject) -> dict[str, Any]:
    if isinstance(value, RushObject):
        return value.to_dict()
    return upload_object(value)


def _config_to_rex(config: SmolSimilarityConfig | None) -> str:
    if config is None:
        return "None"

    return Template(
        """Some (smol_similarity_sumo::SmolSimilarityConfig {
        min_similarity = $min_similarity,
        min_results = $min_results,
        max_results = $max_results
      })"""
    ).substitute(
        min_similarity=optional_str(config.min_similarity),
        min_results=optional_str(config.min_results),
        max_results=optional_str(config.max_results),
    )


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def smol_similarity_sumo(
    smol_partitions: list[Path | str | RushObject],
    input_smis: Path | str | RushObject,
    config: SmolSimilarityConfig | None = None,
    run_spec: RunSpec = RunSpec(cpus=4, gpus=0),
    run_opts: RunOpts = RunOpts(),
) -> Run[ResultRef]:
    """Submit a smol-similarity run.

    Args:
        smol_partitions: JSON object files, each containing a list of library SMILES.
        input_smis: JSON object file containing query SMILES.
        config: Optional search controls.

    Returns:
        A :class:`~rush.runs.Run` handle. Call ``.collect()`` for :class:`ResultRef`,
        then ``.fetch()`` or ``.save()``.
    """

    partition_objects = [_to_uploaded_json_object(partition) for partition in smol_partitions]
    input_smis_object = _to_uploaded_json_object(input_smis)

    partition_objects_rex = ", ".join(
        f'VirtualObject {{ path = "{obj["path"]}", format = ObjectFormat::json, size = 0 }}'
        for obj in partition_objects
    )

    rex = Template("""let
  cfg = $config,
  obj_j = λ j →
    VirtualObject { path = j, format = ObjectFormat::json, size = 0 }
in
  smol_similarity_sumo_s
    ($run_spec)
    cfg
    [$smol_partition_objects]
    (obj_j \"$input_smis_path\")
""").substitute(
        run_spec=run_spec._to_rex(),
        config=_config_to_rex(config),
        smol_partition_objects=partition_objects_rex,
        input_smis_path=input_smis_object["path"],
    )

    try:
        return Run(_submit_rex(rex, run_opts), ResultRef)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
