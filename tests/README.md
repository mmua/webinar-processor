# Webinar Processor Test Suite

This directory contains the test suite for the Webinar Processor application. The tests are organized to validate different aspects of the system, from individual components to complete workflows.

## Test Organization

- **Unit Tests**: Test individual functions and components in isolation
  - `/tests/utils/` - Tests for utility functions
  - `/tests/test_split_text.py` - Tests for text processing utilities

- **Command Tests**: Test CLI commands
  - `/tests/commands/` - Tests for individual CLI commands

- **Integration Tests**: Test workflows and component interactions
  - `/tests/integration/` - Tests for end-to-end workflows

## Verification Strategy

Our test verification strategy is designed to ensure the reliability and correctness of the Webinar Processor. The strategy includes:

### 1. Multi-level Testing

- **Unit Testing**: Verify individual components function correctly
- **Integration Testing**: Verify components work together as expected
- **Command Testing**: Verify CLI interface behaves correctly

### 2. Test Fixtures

Common fixtures in `conftest.py` provide:
- Mock objects for external dependencies (YouTube, OpenAI, etc.)
- Sample data (transcripts, audio files, etc.)
- Temporary directories for file operations

### 3. Dependency Handling

- External dependencies are mocked to avoid network calls and resource-intensive operations
- Fallbacks are provided for dependencies that might not be installed
- Tests are designed to run in isolated environments

### 4. Testing Guidelines

Each test should:
- Have a clear docstring explaining what is being tested
- Include explicit assertions with helpful error messages
- Handle setup and teardown properly
- Mock external dependencies appropriately
- Cover both success and error cases

## Running Tests

To run the tests:

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test module
python -m pytest tests/utils/test_path.py

# Run a specific test
python -m pytest tests/utils/test_path.py::test_ensure_dir_exists

# Run with coverage report
python -m pytest --cov=webinar_processor
```

## Adding New Tests

When adding new tests:

1. Follow the existing directory structure
2. Include detailed docstrings with verification strategy
3. Use existing fixtures when possible
4. Mock external dependencies
5. Test both success and error cases 