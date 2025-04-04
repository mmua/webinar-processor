"""
Webinar Processor Test Configuration
====================================

This file contains common fixtures and configuration for the test suite.

Test Verification Strategy
-------------------------
The test suite follows a multi-level verification strategy:

1. Unit Tests:
   - Test individual functions and components in isolation
   - Mock external dependencies to ensure tests are fast and reliable
   - Focus on clear input/output validation and edge cases

2. Integration Tests:
   - Test workflows that combine multiple components
   - Validate the interaction between different modules
   - Use controlled environment mocks for external services

3. Test Fixtures:
   - Provide common test data and mock objects
   - Ensure consistent test environment across test modules
   - Reduce code duplication and improve maintainability

4. Dependency Handling:
   - Use dependency injection for better testability
   - Mock external dependencies that are unavailable or slow
   - Provide fallbacks for dependencies that might not be installed

5. Coverage Goals:
   - Aim for high coverage of core business logic
   - Focus on the most critical and complex code paths
   - Include both success cases and error handling

For each test, the verification process should include:
- Clear documentation of what is being tested
- Explicit assertions with helpful error messages
- Mocking of external dependencies when appropriate
- Testing of edge cases and error conditions
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path

@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content="Test response"
                )
            )
        ]
    )
    return mock

@pytest.fixture
def mock_youtube():
    """Mock YouTube download functionality."""
    mock = MagicMock()
    stream_mock = MagicMock()
    stream_mock.download.return_value = "test_video.mp4"
    
    streams_mock = MagicMock()
    streams_mock.filter.return_value = streams_mock
    streams_mock.order_by.return_value = streams_mock
    streams_mock.desc.return_value = streams_mock
    streams_mock.first.return_value = stream_mock
    
    mock.streams = streams_mock
    mock.thumbnail_url = "https://example.com/thumbnail.jpg"
    return mock

@pytest.fixture
def mock_sbert_punc():
    """Mock SbertPuncCase for text punctuation."""
    mock = MagicMock()
    mock.punctuate.return_value = "Test punctuated text."
    return mock

@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    audio_path = temp_dir / "test_audio.wav"
    audio_path.touch()
    return audio_path

@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file for testing."""
    video_path = temp_dir / "test_video.mp4"
    video_path.touch()
    return video_path

@pytest.fixture
def sample_transcript():
    """Return a sample transcript for testing."""
    return [
        {"start": 0.0, "end": 2.0, "text": "Hello, this is a test."},
        {"start": 2.0, "end": 4.0, "text": "This is another test."}
    ]

@pytest.fixture(autouse=True)
def mock_external_deps(mock_sbert_punc):
    """Automatically mock external dependencies for all tests."""
    with patch('webinar_processor.utils.openai.IdentityPuncCase', return_value=mock_sbert_punc):
        yield
