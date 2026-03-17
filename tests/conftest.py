"""Shared pytest configuration: auto-mark slow tests and skip when queues are busy."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import requests

# Test files that don't submit jobs to the Rush API — always safe to run.
FAST_TEST_PREFIXES = (
    "test_convert_",
    "test_merge",
    "test_fetch_runs",
    "test_client_collect_run",
    "test_exess_output_helpers",
    "test_nnxtb_output_helpers",
    "test_auto3d_output_helpers",
)

# Threshold: if any target has more than this many queued+admitted jobs, skip slow tests.
QUEUE_BUSY_THRESHOLD = 2


@dataclass
class QueueTarget:
    target: str
    created_count: int
    admitted_count: int
    queued_count: int
    running_count: int


def queue_status() -> list[QueueTarget]:
    """Query the Rush GraphQL API for current queue status."""
    endpoint = os.environ.get(
        "RUSH_ENDPOINT",
        "https://tengu-server-prod-api-519406798674.asia-southeast1.run.app",
    )
    token = os.environ.get("RUSH_TOKEN")
    if not token:
        raise RuntimeError("RUSH_TOKEN not set")

    resp = requests.post(
        endpoint,
        json={
            "query": """
                query { queue_status { target created_count admitted_count queued_count running_count } }
            """
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    return [QueueTarget(**entry) for entry in data["data"]["queue_status"]]


def _queues_are_busy() -> bool:
    """Return True if any target queue has more than QUEUE_BUSY_THRESHOLD queued+admitted jobs."""
    try:
        statuses = queue_status()
    except Exception:
        # If we can't reach the API, assume busy and skip slow tests gracefully.
        return True

    for s in statuses:
        if s.queued_count + s.admitted_count > QUEUE_BUSY_THRESHOLD:
            return True
    return False


def _is_fast_test(item: pytest.Item) -> bool:
    """Check if a test item belongs to a fast (non-API-submitting) test file."""
    filename = item.path.name if item.path else ""
    return any(filename.startswith(prefix) for prefix in FAST_TEST_PREFIXES)


# Cache the queue check result for the session.
_queue_busy: bool | None = None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-apply 'slow' marker and longer timeout to API-submitting tests."""
    slow_marker = pytest.mark.slow
    slow_timeout = pytest.mark.timeout(600)

    for item in items:
        if not _is_fast_test(item):
            item.add_marker(slow_marker)
            if not any(m.name == "timeout" for m in item.iter_markers()):
                item.add_marker(slow_timeout)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip slow tests when queues are busy (unless --run-slow-force is passed)."""
    global _queue_busy

    if "slow" not in item.keywords:
        return

    # Allow forcing slow tests to run regardless of queue status.
    if item.config.getoption("--run-slow-force", default=False):
        return

    if _queue_busy is None:
        _queue_busy = _queues_are_busy()
        if _queue_busy:
            print("\n⚠ Rush queues are busy — skipping slow tests")

    if _queue_busy:
        pytest.skip(
            "Rush queues are busy (queued+admitted > %d)" % QUEUE_BUSY_THRESHOLD
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --run-slow-force option to force slow tests even when queues are busy."""
    parser.addoption(
        "--run-slow-force",
        action="store_true",
        default=False,
        help="Run slow tests even when Rush queues are busy",
    )
