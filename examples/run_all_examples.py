#!/usr/bin/env python3
"""
Script to run all examples in the rush-py examples directory.

This script will:
1. Discover all Python example files in subdirectories
2. Run each example sequentially
3. Report success/failure for each example

Usage:
    python run_all_examples.py

Options:
    --continue-on-error: Continue running remaining examples even if one fails
    --quiet, -q: Suppress output from examples (only show summary)
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
# Set UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    # Set environment variable for subprocesses
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # Try to set stdout/stderr encoding (Python 3.7+)
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # Fallback if reconfigure is not available
        pass


def find_example_files(examples_dir: Path) -> list[Path]:
    """Find all Python example files in subdirectories."""
    example_files = []
    for subdir in examples_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            for py_file in subdir.glob('*.py'):
                # Skip this script itself
                if py_file.name != 'run_all_examples.py':
                    example_files.append(py_file)
    # Sort by filename for consistent ordering
    return sorted(example_files, key=lambda p: p.name)


def run_example(example_path: Path, show_output: bool = True) -> tuple[bool, str]:
    """
    Run a single example script.
    
    Args:
        example_path: Path to the example script
        show_output: If True, show output in real-time. If False, capture it.
    
    Returns:
        (success: bool, error_msg: str)
    """
    print(f"\n{'='*70}")
    print(f"Running: {example_path.name}")
    print(f"Path: {example_path}")
    print(f"{'='*70}\n")
    
    # Change to the example's directory so relative paths work
    cwd = example_path.parent
    
    # Prepare environment with UTF-8 encoding
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        if show_output:
            # Show output in real-time by not capturing stdout
            # Still capture stderr for error reporting
            result = subprocess.run(
                [sys.executable, str(example_path)],
                cwd=cwd,
                env=env,
                stdout=None,  # Don't capture - show in real-time
                stderr=subprocess.PIPE,  # Capture for error reporting
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600  # 10 minute timeout per example
            )
        else:
            # Capture both for quiet mode
            result = subprocess.run(
                [sys.executable, str(example_path)],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=600
            )
        
        if result.returncode == 0:
            print(f"\n✓ Successfully completed: {example_path.name}")
            return True, ""
        else:
            error_msg = result.stderr if result.stderr else "Unknown error"
            print(f"\n✗ Failed: {example_path.name}")
            if not show_output and result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if error_msg:
                print("STDERR:")
                print(error_msg)
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "Example timed out after 10 minutes"
        print(f"✗ Timeout: {example_path.name}")
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Exception running example: {str(e)}"
        print(f"✗ Error: {example_path.name}")
        print(error_msg)
        return False, error_msg


def main():
    parser = argparse.ArgumentParser(
        description="Run all examples in the rush-py examples directory"
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Continue running remaining examples even if one fails'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output from examples (only show summary)'
    )
    args = parser.parse_args()
    
    # Get the examples directory (parent of this script)
    examples_dir = Path(__file__).parent
    
    print(f"Discovering examples in: {examples_dir}")
    example_files = find_example_files(examples_dir)
    
    if not example_files:
        print("No example files found!")
        return 1
    
    print(f"\nFound {len(example_files)} example(s):")
    for ex in example_files:
        print(f"  - {ex.relative_to(examples_dir)}")
    
    print(f"\n{'='*70}")
    print("Starting to run examples...")
    print(f"{'='*70}")
    
    results = []
    for example_path in example_files:
        success, error = run_example(example_path, show_output=not args.quiet)
        results.append((example_path, success, error))
        
        if not success and not args.continue_on_error:
            print(f"\n{'='*70}")
            print("Stopping due to error (use --continue-on-error to continue)")
            print(f"{'='*70}")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    print(f"Total examples: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\n✓ Successful examples:")
        for path, _, _ in successful:
            print(f"  - {path.name}")
    
    if failed:
        print("\n✗ Failed examples:")
        for path, _, error in failed:
            print(f"  - {path.name}")
            if error and args.quiet:
                print(f"    Error: {error[:100]}...")
    
    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
