"""
MMseqs2 module for the Rush Python client.

MMseqs2 generates multiple-sequence alignments (MSAs) from amino acid
sequences.

Usage::

    from rush import mmseqs2

    paths = mmseqs2.search(["MKFLILLFNILCL..."]).save()
"""

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Literal, NewType, Self

from gql.transport.exceptions import TransportQueryError

from ._rex import optional_str
from .objects import RushObject
from .runs import Run, RunOpts, RunSpec
from .session import _submit_rex

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


Result = NewType("Result", list[str])
"""Parsed MMseqs2 results: one A3M text per input sequence."""

ResultPaths = NewType("ResultPaths", list[Path])
"""Workspace paths for saved MMseqs2 A3M files."""


@dataclass(frozen=True)
class ResultRef:
    """Lightweight reference to MMseqs2 outputs in the Rush object store."""

    msas: list[RushObject]

    def __getitem__(self, index: int) -> RushObject:
        return self.msas[index]

    def __len__(self) -> int:
        return len(self.msas)

    def __iter__(self) -> Iterator[RushObject]:
        return iter(self.msas)

    @classmethod
    def from_raw_output(cls, res: Any) -> Self:
        """Parse raw ``collect_run`` output into a ``ResultRef``."""
        if not isinstance(res, list) or len(res) == 0:
            raise ValueError(
                f"mmseqs2 output received unexpected format: {type(res).__name__}"
            )
        # collect_run returns [[dict, ...], ...] (nested per sequence)
        # or [dict, ...] (flattened for single sequence)
        items = res
        if len(items) > 0 and isinstance(items[0], list):
            # Nested: flatten all sublists into one list
            items = [obj for sublist in items for obj in sublist]
        return cls(
            msas=[RushObject.from_dict(obj) for obj in items],
        )

    def fetch(self) -> Result:
        """Download MMseqs2 outputs and parse into A3M strings."""
        a3ms: list[str] = []
        for obj in self.msas:
            a3ms.append(obj.fetch_bytes().decode())
        return Result(a3ms)

    def save(self) -> ResultPaths:
        """Download MMseqs2 outputs and save as A3M files to the workspace."""
        return ResultPaths([obj.save(ext="a3m") for obj in self.msas])


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def search(
    sequences: list[str],
    prefilter_mode: Literal["KMer", "Ungapped", "Exhaustive"] | None = None,
    sensitivity: float | None = None,
    expand_eval: float | None = None,
    align_eval: int | None = None,
    diff: int | None = None,
    qsc: float | None = None,
    max_accept: int | None = None,
    run_spec: RunSpec = RunSpec(gpus=1),
    run_opts: RunOpts = RunOpts(),
) -> Run[ResultRef]:
    """
    Submit an MMseqs2 sequence search for the given amino acid *sequences*.

    Returns a :class:`~rush.runs.Run` handle. Call ``.fetch()`` to get the
    parsed A3M results, or ``.save()`` to write them to disk.
    """

    # TODO: set use_upstream_server to `None` for prod, when it works again
    rex = Template("""
mmseqs2_rex_s
  ($run_spec)
  (mmseqs2_rex::Mmseqs2Config {
    prefilter_mode = $maybe_prefilter_mode,
    sensitivity = $maybe_sensitivity,
    expand_eval = $maybe_expand_eval,
    align_eval = $maybe_align_eval,
    diff = $maybe_diff,
    qsc = $maybe_qsc,
    max_accept = $maybe_max_accept,
    use_upstream_server = (Some "yes")
  })
  $sequences
""").substitute(
        run_spec=run_spec._to_rex(),
        maybe_prefilter_mode=optional_str(prefilter_mode),
        maybe_sensitivity=optional_str(sensitivity),
        maybe_expand_eval=optional_str(expand_eval),
        maybe_align_eval=optional_str(align_eval),
        maybe_diff=optional_str(diff),
        maybe_qsc=optional_str(qsc),
        maybe_max_accept=optional_str(max_accept),
        sequences=f"[\n        {',\n        '.join([f'"{seq}"' for seq in sequences])}]",
    )
    try:
        return Run(_submit_rex(rex, run_opts), ResultRef)

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise
