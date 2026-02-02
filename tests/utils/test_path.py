"""
Path Utility Tests
=================

This module contains tests for path-related utility functions.

Test Verification Strategy
-------------------------
- Verify that directory creation functions work correctly
- Test both new and existing directories to ensure idempotence
- Use temporary directories to ensure tests don't modify real files
- Verify proper error handling for invalid paths

Each test function includes:
- Descriptive docstring explaining what is being tested
- Setup of necessary test environment
- Verification of expected outcomes
- Cleanup of test resources when needed
"""

import pytest
from pathlib import Path
from webinar_processor.utils.path import ensure_dir_exists

def test_ensure_dir_exists(temp_dir):
    """
    Test that ensure_dir_exists creates directory if it doesn't exist.
    
    Verification:
    1. Directory should be created when it doesn't exist
    2. The Path object should exist and be a directory after the call
    """
    test_dir = temp_dir / "test_dir"
    ensure_dir_exists(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()

def test_ensure_dir_exists_existing(temp_dir):
    """
    Test that ensure_dir_exists works with existing directory.
    
    Verification:
    1. No exceptions should be raised when directory already exists
    2. The existing directory should remain intact
    """
    test_dir = temp_dir / "test_dir"
    test_dir.mkdir()
    ensure_dir_exists(test_dir)  # Should not raise an exception
    assert test_dir.exists()
    assert test_dir.is_dir() 