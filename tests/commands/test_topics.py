"""
Topics Command Tests
==================

This module contains tests for the topic extraction command.

Test Verification Strategy
-------------------------
- Verify that topic extraction commands function correctly
- Test configuration file loading and path resolution
- Mock OpenAI API calls to isolate tests
- Verify proper CLI output and error handling
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from webinar_processor import cli
from webinar_processor.utils.package import get_config_path

def test_config_paths_exist():
    """
    Test that the configuration files needed by the topics command exist.
    
    Verification:
    1. The intermediate prompt file path should be resolvable
    2. The final prompt file path should be resolvable
    3. Both files should exist and be readable
    
    This test ensures that the paths used in the topics command are valid.
    """
    # Get the paths for the prompt files
    intermediate_prompt_path = get_config_path("intermediate-topics-prompt.txt")
    final_prompt_path = get_config_path("final-topics-prompt-onepass.txt")
    
    # Check that the paths exist
    assert os.path.exists(intermediate_prompt_path), f"Intermediate prompt file not found at {intermediate_prompt_path}"
    assert os.path.exists(final_prompt_path), f"Final prompt file not found at {final_prompt_path}"
    
    # Check that the files are readable
    with open(intermediate_prompt_path, "r", encoding="utf-8") as f:
        intermediate_content = f.read()
        assert len(intermediate_content) > 0, "Intermediate prompt file is empty"
    
    with open(final_prompt_path, "r", encoding="utf-8") as f:
        final_content = f.read()
        assert len(final_content) > 0, "Final prompt file is empty"

def test_topics_command_mock(mock_openai, temp_dir):
    """
    Test the topics command functionality with mocked OpenAI responses.

    Verification:
    1. Command should accept a text file input
    2. Command should call the TopicExtractionService
    3. Command should output the extracted topics

    This test mocks the service to avoid actual API calls.
    """
    # Create a sample text file
    test_file = temp_dir / "test_transcript.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("This is a test transcript about artificial intelligence and machine learning topics.")

    # Set up the runner
    runner = CliRunner()

    # Mock the TopicExtractionService
    with patch.dict('os.environ', {'LLM_API_KEY': 'fake-api-key'}):
        with patch('webinar_processor.commands.cmd_topics.TopicExtractionService') as MockService:
            mock_service = MockService.return_value
            mock_service.extract_topics.return_value = ("Intermediate topics: AI, ML", "Final topics: Artificial Intelligence, Machine Learning")

            # Run the command
            result = runner.invoke(
                cli,
                ['topics', str(test_file)]
            )

            # Check that the command ran successfully
            assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}: {result.output}"

            # Check that the mocked service was called
            mock_service.extract_topics.assert_called_once()

            # Check that the output contains the expected topics
            assert "Final topics" in result.output, f"Expected topics in output, got: {result.output}"