
# Run test
import sys

from rush_py2.auto3d import auto3d


if __name__ == "__main__":
    result = auto3d(["C1=CC=CC=C1"])
    print(result)