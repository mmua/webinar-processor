"""
Tests for the LLM module (LLMConfig and LLMClient).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestLLMConfig:
    """Test cases for LLMConfig class."""

    def test_get_api_key_from_llm_api_key(self):
        """Test that LLM_API_KEY is preferred."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'llm-key', 'OPENAI_API_KEY': 'openai-key'}):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_api_key() == 'llm-key'

    def test_get_api_key_fallback_to_openai(self):
        """Test fallback to OPENAI_API_KEY for backwards compatibility."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'openai-key'}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_api_key() == 'openai-key'

    def test_get_api_key_none_when_missing(self):
        """Test that None is returned when no API key is set."""
        with patch.dict('os.environ', {}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_api_key() is None

    def test_get_base_url_default(self):
        """Test default base URL."""
        with patch.dict('os.environ', {}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_base_url() == 'https://api.openai.com/v1'

    def test_get_base_url_custom(self):
        """Test custom base URL from environment."""
        with patch.dict('os.environ', {'LLM_BASE_URL': 'https://custom.api.com/v1'}):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_base_url() == 'https://custom.api.com/v1'

    def test_get_model_default_for_task(self):
        """Test getting default model for a known task."""
        with patch.dict('os.environ', {}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_model('summarization') == 'gpt-5-mini'
            assert LLMConfig.get_model('topics') == 'gpt-5.2'
            assert LLMConfig.get_model('quiz') == 'gpt-5.2'

    def test_get_model_from_task_specific_env_var(self):
        """Test that task-specific env var takes precedence."""
        with patch.dict('os.environ', {'LLM_SUMMARIZATION_MODEL': 'custom-model'}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_model('summarization') == 'custom-model'

    def test_get_model_from_default_env_var(self):
        """Test that LLM_DEFAULT_MODEL is used when task-specific is not set."""
        with patch.dict('os.environ', {'LLM_DEFAULT_MODEL': 'default-model'}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_model('unknown_task') == 'default-model'

    def test_get_model_unknown_task_hardcoded_default(self):
        """Test fallback to hardcoded default for unknown task."""
        with patch.dict('os.environ', {}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_model('unknown_task') == 'gpt-5-mini'

    def test_get_model_env_var_priority(self):
        """Test priority: task-specific > LLM_DEFAULT_MODEL > hardcoded."""
        # Task-specific should win
        with patch.dict('os.environ', {
            'LLM_QUIZ_MODEL': 'task-specific',
            'LLM_DEFAULT_MODEL': 'default'
        }, clear=True):
            from webinar_processor.llm.config import LLMConfig
            assert LLMConfig.get_model('quiz') == 'task-specific'

    def test_validate_success(self):
        """Test validate passes when API key is set."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            from webinar_processor.llm.config import LLMConfig
            LLMConfig.validate()  # Should not raise

    def test_validate_with_openai_key(self):
        """Test validate passes with legacy OPENAI_API_KEY."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            LLMConfig.validate()  # Should not raise

    def test_validate_raises_when_no_key(self):
        """Test validate raises ValueError when no API key is set."""
        with patch.dict('os.environ', {}, clear=True):
            from webinar_processor.llm.config import LLMConfig
            with pytest.raises(ValueError) as exc_info:
                LLMConfig.validate()
            assert 'LLM_API_KEY' in str(exc_info.value)
            assert 'OPENAI_API_KEY' in str(exc_info.value)


class TestLLMClient:
    """Test cases for LLMClient class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_init_success(self, mock_openai_client):
        """Test successful client initialization."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                assert client.client is not None

    def test_init_raises_without_api_key(self):
        """Test that init raises ValueError without API key."""
        with patch.dict('os.environ', {}, clear=True):
            from webinar_processor.llm.client import LLMClient
            with pytest.raises(ValueError):
                LLMClient()

    def test_generate_success(self, mock_openai_client):
        """Test successful text generation."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                result = client.generate("Test prompt")

                assert result == "Test response"
                mock_openai_client.chat.completions.create.assert_called_once()

    def test_generate_with_model_override(self, mock_openai_client):
        """Test generation with explicit model."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                client.generate("Test prompt", model="gpt-4")

                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]['model'] == 'gpt-4'

    def test_generate_with_parameters(self, mock_openai_client):
        """Test that parameters are passed correctly."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                client.generate("Test prompt", max_tokens=200)

                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]['max_completion_tokens'] == 200

    def test_generate_api_error_raises_llmerror(self, mock_openai_client):
        """Test that API errors raise LLMError."""
        from webinar_processor.llm.exceptions import LLMError
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                with pytest.raises(LLMError) as exc_info:
                    client.generate("Test prompt")

                assert "LLM generation failed" in str(exc_info.value)

    def test_extract_speaker_name_success(self, mock_openai_client):
        """Test successful speaker name extraction."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "John Smith"

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                result = client.extract_speaker_name("Hi, I'm John Smith")

                assert result == "John Smith"

    def test_extract_speaker_name_with_quotes(self, mock_openai_client):
        """Test extraction strips quotes from response."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = '"Jane Doe"'

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                result = client.extract_speaker_name("My name is Jane Doe")

                assert result == "Jane Doe"

    def test_extract_speaker_name_with_punctuation(self, mock_openai_client):
        """Test extraction strips trailing punctuation."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = "Bob Johnson."

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                result = client.extract_speaker_name("This is Bob Johnson")

                assert result == "Bob Johnson"

    @pytest.mark.parametrize("response", ["None", "none", "NONE", "null", "NULL", "n/a", "N/A", ""])
    def test_extract_speaker_name_none_responses(self, mock_openai_client, response):
        """Test that none-like responses return None."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = response

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                result = client.extract_speaker_name("No introduction here")

                assert result is None

    def test_extract_speaker_name_generate_returns_none(self, mock_openai_client):
        """Test when generate returns None (service unavailable)."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                result = client.extract_speaker_name("Hi, I'm Alice")

                assert result is None

    def test_extract_speaker_name_empty_after_strip(self, mock_openai_client):
        """Test when name becomes empty after stripping punctuation."""
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = '""'

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                result = client.extract_speaker_name("Some text")

                assert result is None

    def test_extract_speaker_name_prompt_structure(self, mock_openai_client):
        """Test that the prompt contains expected elements."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                client.extract_speaker_name("Test text content")

                call_args = mock_openai_client.chat.completions.create.call_args
                prompt = call_args[1]['messages'][0]['content']

                assert "extract" in prompt.lower()
                assert "speaker" in prompt.lower()
                assert "name" in prompt.lower()
                assert "Test text content" in prompt

    def test_extract_speaker_name_parameters(self, mock_openai_client):
        """Test that extract_speaker_name uses correct parameters."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()
                client.extract_speaker_name("Test text")

                call_args = mock_openai_client.chat.completions.create.call_args
                assert call_args[1]['max_completion_tokens'] == 50

    def test_generate_token_limit_exceeded(self, mock_openai_client):
        """Test that TokenLimitError is raised when token limit is exceeded."""
        from webinar_processor.llm.exceptions import TokenLimitError

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                with patch('webinar_processor.llm.client._count_tokens', return_value=200000):
                    from webinar_processor.llm.client import LLMClient
                    client = LLMClient()

                    # Mock a prompt that exceeds token limit (200000 tokens > 128000 limit)
                    with pytest.raises(TokenLimitError) as exc_info:
                        client.generate("Test prompt", max_tokens=100)

                    assert "exceeds token limit" in str(exc_info.value)

    def test_generate_within_token_limit(self, mock_openai_client):
        """Test that generate works normally when within token limit."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                from webinar_processor.llm.client import LLMClient
                client = LLMClient()

                # Short prompt should be fine
                result = client.generate("Short prompt", max_tokens=100)

                assert result == "Response"
                mock_openai_client.chat.completions.create.assert_called_once()
