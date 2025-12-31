import sys

from rush_py2.client import fetch_runs

if __name__ == "__main__":
    print(fetch_runs(name_contains="Rush-Py Test"), file=sys.stderr)
