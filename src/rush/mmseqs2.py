#!/usr/bin/env python3
import sys
from string import Template
from typing import Literal

from gql.transport.exceptions import TransportQueryError

from .client import (
    PROJECT_ID,
    RunOpts,
    RunSpec,
    _submit_rex,
    collect_run,
)
from .utils import optional_str


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
    collect=False,
):
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
        run_id = _submit_rex(PROJECT_ID, rex, run_opts)
        if collect:
            return collect_run(run_id)
        else:
            return run_id

    except TransportQueryError as e:
        if e.errors:
            for error in e.errors:
                print(f"Error: {error['message']}", file=sys.stderr)
