import sys

import pytest

from rush import fetch_run_info, fetch_runs


def test_fetch_runs():
    runs = fetch_runs(name_contains="Rush-Py Test")
    if not runs:
        pytest.skip("No runs found matching 'Rush-Py Test'.")
    print(fetch_run_info(runs[0]), file=sys.stderr)


if __name__ == "__main__":
    test_fetch_runs()
