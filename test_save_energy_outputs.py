#!/usr/bin/env python3
"""
Test script for save_energy_outputs() function logic.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add the src directory to the path
sys.path.insert(0, '/home/claw/rush-py/src')

from rush.exess import save_energy_outputs
from rush.client import RunError

def test_single_output():
    """Test single output case (only JSON, no HDF5)."""
    print("Test 1: Single output (JSON only)...")
    with patch('rush.exess.save_object') as mock_save:
        mock_save.return_value = Path('/tmp/output.json')
        
        res = ({'path': 'json_path'},)
        result = save_energy_outputs(res)
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
        json_path, hdf5_path = result
        assert json_path == Path('/tmp/output.json'), f"Expected Path, got {json_path}"
        assert hdf5_path is None, f"Expected None for hdf5_path, got {hdf5_path}"
        print("✓ Test 1 passed: Single output returns (json_path, None)")

def test_two_outputs_with_json():
    """Test two outputs with Json key in res[1]."""
    print("\nTest 2: Two outputs with Json key...")
    with patch('rush.exess.save_object') as mock_save:
        mock_save.side_effect = [Path('/tmp/output.json'), Path('/tmp/output_json.json')]
        
        res = (
            {'path': 'json_path'},
            {'Json': {'path': 'json_converted_path'}}
        )
        result = save_energy_outputs(res)
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
        json_path, json_converted_path = result
        assert json_path == Path('/tmp/output.json')
        assert json_converted_path == Path('/tmp/output_json.json')
        print("✓ Test 2 passed: Two outputs with Json key returns (json_path, json_converted_path)")

def test_two_outputs_with_hdf5_success():
    """Test two outputs with Hdf5 key and successful extraction."""
    print("\nTest 3: Two outputs with Hdf5 key (successful extraction)...")
    with patch('rush.exess.save_object') as mock_save:
        mock_save.side_effect = [Path('/tmp/output.json'), Path('/tmp/output.hdf5')]
        
        res = (
            {'path': 'json_path'},
            {'Hdf5': {'path': 'hdf5_path'}}
        )
        result = save_energy_outputs(res)
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
        json_path, hdf5_path = result
        assert json_path == Path('/tmp/output.json')
        assert hdf5_path == Path('/tmp/output.hdf5')
        print("✓ Test 3 passed: Two outputs with Hdf5 key returns (json_path, hdf5_path)")

def test_two_outputs_with_hdf5_empty():
    """Test two outputs with Hdf5 key but empty HDF5 (ValueError with 'only directories')."""
    print("\nTest 4: Two outputs with Hdf5 key (empty HDF5)...")
    with patch('rush.exess.save_object') as mock_save:
        json_path = Path('/tmp/output.json')
        
        def side_effect(*args, **kwargs):
            if args[0] == 'json_path':
                return json_path
            else:
                raise ValueError("only directories found in tar archive")
        
        mock_save.side_effect = side_effect
        
        res = (
            {'path': 'json_path'},
            {'Hdf5': {'path': 'hdf5_path'}}
        )
        result = save_energy_outputs(res)
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
        json_path_result, hdf5_path = result
        assert json_path_result == json_path
        assert hdf5_path is None, f"Expected None for hdf5_path when empty, got {hdf5_path}"
        print("✓ Test 4 passed: Two outputs with empty HDF5 returns (json_path, None)")

def test_run_error():
    """Test RunError case."""
    print("\nTest 5: RunError input...")
    error = RunError("Test error")
    result = save_energy_outputs(error)
    
    assert result is error, "Expected RunError to be returned as-is"
    print("✓ Test 5 passed: RunError is returned as-is")

def test_list_input():
    """Test that list input is converted to tuple."""
    print("\nTest 6: List input conversion...")
    with patch('rush.exess.save_object') as mock_save:
        mock_save.return_value = Path('/tmp/output.json')
        
        res = [{'path': 'json_path'}]  # List instead of tuple
        result = save_energy_outputs(res)
        
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected tuple of length 2, got {len(result)}"
        json_path, hdf5_path = result
        assert json_path == Path('/tmp/output.json')
        assert hdf5_path is None
        print("✓ Test 6 passed: List input is converted to tuple and returns (json_path, None)")

if __name__ == '__main__':
    print("Running tests for save_energy_outputs()...\n")
    test_single_output()
    test_two_outputs_with_json()
    test_two_outputs_with_hdf5_success()
    test_two_outputs_with_hdf5_empty()
    test_run_error()
    test_list_input()
    print("\n✓ All tests passed!")
