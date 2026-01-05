import json
import sys
from pathlib import Path

from rush.convert.json import to_json
from rush.convert.pdb import from_pdb


def normalize_json(obj):
    """Recursively normalize JSON to ignore key ordering."""
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    else:
        return obj


def test_pdb_conversion():
    """Test that all .pdb files convert to JSON matching their .pdb.json counterparts."""
    test_cache_dir = Path(__file__).parent.parent / "test_cache"

    # Find all .pdb files
    pdb_files = sorted(test_cache_dir.glob("*.pdb"))

    if not pdb_files:
        print("No .pdb files found in test_cache directory", file=sys.stderr)
        return

    print(f"Found {len(pdb_files)} PDB files to test", file=sys.stderr)

    passed = 0
    failed = 0

    for pdb_file in pdb_files:
        json_file = pdb_file.with_suffix(pdb_file.suffix + ".json")

        if not json_file.exists():
            print(
                f"SKIP: {pdb_file.name} - no corresponding .pdb.json file found",
                file=sys.stderr,
            )
            continue

        # Read and convert PDB file
        with open(pdb_file, "r") as f:
            pdb_contents = f.read()

        try:
            # Convert PDB to TRC structures
            trcs = from_pdb(pdb_contents)

            # Convert TRC structures to JSON
            converted_json_str = to_json(trcs)
            converted_json = json.loads(converted_json_str)

            # Read expected JSON
            with open(json_file, "r") as f:
                expected_json = json.load(f)

            # Normalize both JSON objects (to ignore key ordering)
            normalized_converted = normalize_json(converted_json)
            normalized_expected = normalize_json(expected_json)

            # Compare
            if normalized_converted == normalized_expected:
                print(f"PASS: {pdb_file.name}", file=sys.stderr)
                passed += 1
            else:
                print(f"FAIL: {pdb_file.name} - JSON does not match", file=sys.stderr)
                failed += 1

                # Show detailed diff for debugging
                print(
                    f"  Expected keys: {set(normalized_expected[0].keys()) if normalized_expected else set()}",
                    file=sys.stderr,
                )
                print(
                    f"  Converted keys: {set(normalized_converted[0].keys()) if normalized_converted else set()}",
                    file=sys.stderr,
                )

        except Exception as e:
            print(f"ERROR: {pdb_file.name} - {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Results: {passed} passed, {failed} failed", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    return failed == 0


# Run test
if __name__ == "__main__":
    success = test_pdb_conversion()
    sys.exit(0 if success else 1)
