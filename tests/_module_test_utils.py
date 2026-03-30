"""Shared assertions for end-to-end module tests."""

from typing import TypeVar

from rush.runs import Run

R = TypeVar("R")


def assert_run_collects_and_caches(run: Run[R], expected_ref_type: type[R]) -> R:
    """Assert that `collect()` returns the expected ref type and caches it."""
    ref = run.collect()
    assert isinstance(ref, expected_ref_type)
    assert run.collect() is ref
    return ref
