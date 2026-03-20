#!/usr/bin/env python3
"""
MMseqs2 module helpers for the Rush Python client.

MMseqs2 generates multiple-sequence alignments (MSAs) from amino acid
sequences. The fetched output is the in-memory A3M text for each input
sequence, while the saved output is the corresponding set of `.a3m` files in
the workspace.
"""

import sys
from pathlib import Path
from string import Template
from typing import Any, Literal, overload

from gql.transport.exceptions import TransportQueryError

from .client import (
    RunID,
    RunOpts,
    RunSpec,
    _get_project_id,
    _submit_rex,
    collect_run,
    fetch_object,
    save_object,
)
from .utils import optional_str


@overload
def mmseqs2(
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
    collect: Literal[False] = False,
) -> RunID: ...
@overload
def mmseqs2(
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
    collect: Literal[True] = True,
) -> list[dict[str, Any]]: ...
@overload
def mmseqs2(
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
    collect: bool = False,
) -> list[dict[str, Any]] | RunID: ...


def mmseqs2(
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
    collect: bool = False,
) -> list[dict[str, Any]] | RunID:
    """
    Run MMseqs2 on one or more amino acid sequences.

    The collected result is a list of Rush object-store paths to A3M files, one
    per input sequence.
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
        run_id = _submit_rex(_get_project_id(), rex, run_opts)
        if not collect:
            return run_id

        out = collect_run(run_id)
        assert isinstance(out, list)
        assert len(out) == len(sequences)
        for out_i in out:
            assert isinstance(out_i, list)
        if len(out) == 1:
            out = out[0]
        return out

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
        raise e


def fetch_outputs(res: list[dict[str, Any]]) -> list[str]:
    """
    Fetch MMseqs2 outputs into memory as A3M strings.

    Args:
        res: Collected output from mmseqs2(), one object per input sequence.

    Returns:
        A3M text outputs in the same order as the collected outputs.
    """
    outputs = res
    a3ms: list[str] = []
    for output_obj in outputs:
        a3m = fetch_object(output_obj["path"])
        a3ms.append(a3m.decode() if isinstance(a3m, bytes) else a3m)
    return a3ms


def save_outputs(res: list[dict[str, Any]]) -> list[Path]:
    """
    Save MMseqs2 outputs into the workspace as `.a3m` files.

    Args:
        res: Collected output from mmseqs2(), one object per input sequence.

    Returns:
        Local `.a3m` paths in the same order as the collected outputs.
    """
    outputs = res
    return [
        save_object(output_obj["path"], type="bin", ext="a3m") for output_obj in outputs
    ]
