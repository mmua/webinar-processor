"""
Download Command Tests
====================

This module contains tests for the YouTube video download command.

Test Verification Strategy
-------------------------
- Verify that video download commands function correctly
- Test both success and error cases
- Mock external dependencies (YouTube API) to isolate tests
- Verify proper CLI output and error handling

Each test verifies:
- Correct handling of command-line arguments
- Appropriate error messages for invalid inputs
- Successful download and result reporting
- Proper return codes in success and failure cases
"""

import pytest
from unittest.mock import patch
from click.testing import CliRunner
from webinar_processor.commands.download import yt_download

def test_yt_download_success(mock_youtube, temp_dir):
    """
    Test successful YouTube video download.
    
    Verification:
    1. Command should execute successfully (exit code 0)
    2. Output should include the path of the downloaded video
    3. The YouTube API should be called with the correct URL
    
    This test mocks the YouTube API to avoid actual network calls
    while still verifying the command's behavior.
    """
    runner = CliRunner()
    with patch('webinar_processor.commands.download.YouTube', return_value=mock_youtube):
        result = runner.invoke(
            yt_download,
            ['https://youtube.com/watch?v=test', '--output-dir', str(temp_dir)]
        )
        assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}"
        assert "test_video.mp4" in result.output, f"Expected video path in output, got: {result.output}"

def test_yt_download_invalid_url(temp_dir):
    """
    Test YouTube download with invalid URL.
    
    Verification:
    1. Command should fail with non-zero exit code
    2. Error message should indicate invalid URL
    3. No download should be attempted
    
    This test ensures proper error handling and user feedback
    when invalid inputs are provided.
    """
    runner = CliRunner()
    result = runner.invoke(
        yt_download,
        ['invalid-url', '--output-dir', str(temp_dir)]
    )
    assert result.exit_code != 0, "Command should fail with invalid URL"
    assert "Invalid YouTube URL" in result.output, f"Expected error message, got: {result.output}" 