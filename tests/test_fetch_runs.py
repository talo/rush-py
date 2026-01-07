import sys

from rush.client import fetch_run_info, fetch_runs

if __name__ == "__main__":
    runs = fetch_runs(name_contains="Rush-Py Test")
    print(fetch_run_info(runs[0]), file=sys.stderr)
