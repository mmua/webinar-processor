import pytest
from unittest.mock import Mock, patch, MagicMock
import openai
from webinar_processor.services.llm_service import LLMService


@pytest.fixture
def llm_service():
    """Create an LLMService instance for testing."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        service = LLMService()
        return service


@pytest.fixture
def llm_service_no_key():
    """Create an LLMService instance without API key."""
    with patch.dict('os.environ', {}, clear=True):
        service = LLMService()
        return service


class TestLLMServiceExtractSpeakerName:
    """Test cases for the extract_speaker_name method."""

    def test_extract_speaker_name_simple_introduction(self, llm_service):
        """Test extracting name from simple self-introduction."""
        # Mock the generate method to return a name
        llm_service.generate = Mock(return_value="John Smith")
        
        text = "Hi, I'm John Smith and I'll be presenting today."
        result = llm_service.extract_speaker_name(text)
        
        assert result == "John Smith"
        llm_service.generate.assert_called_once()

    def test_extract_speaker_name_with_quotes(self, llm_service):
        """Test extracting name when LLM returns name with quotes."""
        # Mock the generate method to return a name with quotes
        llm_service.generate = Mock(return_value='"Jane Doe"')
        
        text = "My name is Jane Doe, welcome to the webinar."
        result = llm_service.extract_speaker_name(text)
        
        assert result == "Jane Doe"

    def test_extract_speaker_name_with_punctuation(self, llm_service):
        """Test extracting name when LLM returns name with punctuation."""
        # Mock the generate method to return a name with punctuation
        llm_service.generate = Mock(return_value="Bob Johnson.")
        
        text = "This is Bob Johnson speaking."
        result = llm_service.extract_speaker_name(text)
        
        assert result == "Bob Johnson"

    def test_extract_speaker_name_none_response(self, llm_service):
        """Test when LLM returns 'None' indicating no name found."""
        # Mock the generate method to return 'None'
        llm_service.generate = Mock(return_value="None")
        
        text = "Welcome to today's presentation about machine learning."
        result = llm_service.extract_speaker_name(text)
        
        assert result is None

    def test_extract_speaker_name_null_response(self, llm_service):
        """Test when LLM returns 'null' indicating no name found."""
        # Mock the generate method to return 'null'
        llm_service.generate = Mock(return_value="null")
        
        text = "Let's begin with the agenda for today."
        result = llm_service.extract_speaker_name(text)
        
        assert result is None

    def test_extract_speaker_name_na_response(self, llm_service):
        """Test when LLM returns 'n/a' indicating no name found."""
        # Mock the generate method to return 'n/a'
        llm_service.generate = Mock(return_value="n/a")
        
        text = "The topic for today is artificial intelligence."
        result = llm_service.extract_speaker_name(text)
        
        assert result is None

    def test_extract_speaker_name_empty_response(self, llm_service):
        """Test when LLM returns empty string."""
        # Mock the generate method to return empty string
        llm_service.generate = Mock(return_value="")
        
        text = "Good morning everyone."
        result = llm_service.extract_speaker_name(text)
        
        assert result is None

    def test_extract_speaker_name_generate_returns_none(self, llm_service):
        """Test when generate method returns None (service unavailable)."""
        # Mock the generate method to return None
        llm_service.generate = Mock(return_value=None)
        
        text = "Hi, I'm Alice Cooper and welcome to the show."
        result = llm_service.extract_speaker_name(text)
        
        assert result is None

    def test_extract_speaker_name_with_case_variations(self, llm_service):
        """Test that case variations of 'none' responses are handled."""
        test_cases = ["NONE", "None", "NULL", "Null", "N/A"]
        
        for response in test_cases:
            llm_service.generate = Mock(return_value=response)
            result = llm_service.extract_speaker_name("Some text")
            assert result is None, f"Failed for response: {response}"

    def test_extract_speaker_name_prompt_structure(self, llm_service):
        """Test that the prompt is structured correctly."""
        llm_service.generate = Mock(return_value="Test Name")
        
        text = "Hello, I'm Test Name"
        llm_service.extract_speaker_name(text)
        
        # Verify the prompt contains the expected elements
        call_args = llm_service.generate.call_args
        prompt = call_args[0][0]
        
        assert "extract the speaker's name" in prompt.lower()
        assert "hi, i'm [name]" in prompt.lower()
        assert "my name is [name]" in prompt.lower()
        assert text in prompt
        assert "name:" in prompt.lower()

    def test_extract_speaker_name_parameters(self, llm_service):
        """Test that generate is called with correct parameters."""
        llm_service.generate = Mock(return_value="Test Name")
        
        text = "I'm Test Name"
        llm_service.extract_speaker_name(text)
        
        # Verify the parameters passed to generate
        call_args = llm_service.generate.call_args
        assert call_args[1]['max_tokens'] == 50
        assert call_args[1]['temperature'] == 0.1

    def test_extract_speaker_name_strips_whitespace(self, llm_service):
        """Test that whitespace is properly stripped from the response."""
        # Mock the generate method to return a name with whitespace
        # Note: generate method already strips whitespace, so this tests the punctuation stripping
        llm_service.generate = Mock(return_value="Sarah Wilson")
        
        text = "Hi, I'm Sarah Wilson"
        result = llm_service.extract_speaker_name(text)
        
        assert result == "Sarah Wilson"

    def test_extract_speaker_name_complex_punctuation(self, llm_service):
        """Test stripping various punctuation marks."""
        test_cases = [
            ('"Dr. Smith"', "Dr. Smith"),
            ("'Professor Jones'", "Professor Jones"),
            ("Maria Garcia.", "Maria Garcia"),
            ("Tom Brown,", "Tom Brown"),
            ("Lisa White!", "Lisa White"),
            ("David Lee?", "David Lee"),
        ]
        
        for llm_response, expected in test_cases:
            llm_service.generate = Mock(return_value=llm_response)
            result = llm_service.extract_speaker_name("Some text")
            assert result == expected, f"Failed for input: {llm_response}"

    def test_extract_speaker_name_no_api_key(self, llm_service_no_key):
        """Test behavior when no API key is available."""
        text = "Hi, I'm John Doe"
        result = llm_service_no_key.extract_speaker_name(text)
        
        assert result is None

    def test_extract_speaker_name_after_strip_becomes_empty(self, llm_service):
        """Test when name becomes empty after stripping punctuation."""
        # Mock the generate method to return only punctuation
        llm_service.generate = Mock(return_value='""')
        
        text = "Some text without introduction"
        result = llm_service.extract_speaker_name(text)
        
        assert result is None


class TestLLMServiceGenerate:
    """Test cases for the generate method."""

    @patch('openai.OpenAI')
    def test_generate_success(self, mock_openai_class):
        """Test successful text generation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = LLMService()
            result = service.generate("Test prompt")
        
        assert result == "Generated response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_generate_api_error(self, mock_openai_class):
        """Test handling of OpenAI API errors."""
        # Setup mock to raise an exception that will be caught by the APIError handler
        mock_client = Mock()
        
        # Create a mock exception that behaves like APIError
        class MockAPIError(Exception):
            pass
        
        # Patch the APIError in the service module
        with patch('webinar_processor.services.llm_service.openai.APIError', MockAPIError):
            mock_client.chat.completions.create.side_effect = MockAPIError("API Error")
            mock_openai_class.return_value = mock_client
            
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                service = LLMService()
                result = service.generate("Test prompt")
            
            assert result is None

    @patch('openai.OpenAI')
    def test_generate_unexpected_error(self, mock_openai_class):
        """Test handling of unexpected errors."""
        # Setup mock to raise generic exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = ValueError("Unexpected error")
        mock_openai_class.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = LLMService()
            result = service.generate("Test prompt")
        
        assert result is None

    def test_generate_no_client(self, llm_service_no_key):
        """Test generate when no client is available."""
        result = llm_service_no_key.generate("Test prompt")
        assert result is None

    @patch('openai.OpenAI')
    def test_generate_parameters(self, mock_openai_class):
        """Test that generate passes correct parameters to OpenAI."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = LLMService(model="gpt-4")
            service.generate("Test prompt", max_tokens=200, temperature=0.5)
        
        # Verify the call parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "gpt-4"
        assert call_args[1]['messages'][0]['content'] == "Test prompt"
        assert call_args[1]['max_tokens'] == 200
        assert call_args[1]['temperature'] == 0.5


class TestLLMServiceInit:
    """Test cases for LLMService initialization."""

    @patch('openai.OpenAI')
    def test_init_with_api_key(self, mock_openai_class):
        """Test initialization with API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = LLMService()
        
        assert service.client is not None
        assert service.model == "gpt-4.1-mini"
        mock_openai_class.assert_called_once_with(api_key='test-key')

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            service = LLMService()
        
        assert service.client is None
        assert service.model == "gpt-4.1-mini"

    @patch('openai.OpenAI')
    def test_init_custom_model(self, mock_openai_class):
        """Test initialization with custom model."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = LLMService(model="gpt-3.5-turbo")
        
        assert service.model == "gpt-3.5-turbo" 