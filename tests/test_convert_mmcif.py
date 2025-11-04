import json
import sys
from pathlib import Path

# Add src to path so we can import rush_py2
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rush_py2.convert.json import to_json
from rush_py2.convert.mmcif import from_mmcif


def normalize_json(obj):
    """Recursively normalize JSON to ignore key ordering."""
    if isinstance(obj, dict):
        return {k: normalize_json(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj]
    else:
        return obj


def test_cif_conversion():
    """Test that all .cif files convert to JSON matching their .cif.json counterparts."""
    test_cache_dir = Path(__file__).parent.parent / "test_cache"

    # Find all .cif files
    cif_files = sorted(test_cache_dir.glob("*.cif"))

    if not cif_files:
        print("No .cif files found in test_cache directory", file=sys.stderr)
        return

    print(f"Found {len(cif_files)} PDB files to test", file=sys.stderr)

    passed = 0
    failed = 0

    for cif_file in cif_files:
        json_file = cif_file.with_suffix(cif_file.suffix + ".json")

        if not json_file.exists():
            print(
                f"SKIP: {cif_file.name} - no corresponding .cif.json file found",
                file=sys.stderr,
            )
            continue

        # Read and convert PDB file
        with open(cif_file, "r") as f:
            cif_contents = f.read()

        try:
            # Convert PDB to TRC structures
            trcs = from_mmcif(cif_contents)

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
                print(f"PASS: {cif_file.name}", file=sys.stderr)
                passed += 1
            else:
                print(f"FAIL: {cif_file.name} - JSON does not match", file=sys.stderr)
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
            print(f"ERROR: {cif_file.name} - {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Results: {passed} passed, {failed} failed", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    return failed == 0


# Run test
if __name__ == "__main__":
    success = test_cif_conversion()
    sys.exit(0 if success else 1)
