"""
Package Resource Tests
=====================

This module tests of package resource handling functions.

Test Verification Strategy
-------------------------
- Verify that configuration files can be located in resources
- Verify paths are returned even for non-existent resources
"""

import os
from webinar_processor.utils.package import get_config_path

def test_get_config_path():
    """
    Test the get_config_path function with various config files.
    
    Verification:
    1. It should find existing config files
    2. The returned paths should be absolute
    3. The files should exist at the returned paths
    """
    # Test with known configuration files
    test_files = [
        "storytell-outline-prompt.txt",
        "storytell-section-task.txt",
        "storytell-quiz-prompt.txt"
    ]
    
    for filename in test_files:
        path = get_config_path(filename)
        assert path, f"Failed to get path for {filename}"
        assert os.path.isabs(path), f"Path should be absolute: {path}"
        assert os.path.exists(path), f"File should exist at {path}"

def test_nonexistent_resource():
    """
    Test handling of non-existent resources.
    
    Verification:
    1. When requesting a file that doesn't exist, a path should still be returned
    2. The path should be absolute and point to the expected location
    """
    # Test with a non-existent configuration file
    nonexistent_file = "nonexistent_config.txt"
    path = get_config_path(nonexistent_file)
    
    # Verify that a path is returned
    assert path, f"Should return a path even for non-existent file {nonexistent_file}"
    assert os.path.isabs(path), f"Path should be absolute: {path}"
    
    # Verify that the path follows the expected pattern
    assert 'resources/conf/' in path or 'resources\\conf\\' in path, \
        f"Path should point to resources/conf directory: {path}"
    assert path.endswith(nonexistent_file), \
        f"Path should end with the requested filename: {path}" 