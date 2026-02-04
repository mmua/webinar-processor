"""Tests for TopicExtractionService."""

import pytest
from unittest.mock import patch
from webinar_processor.services.topic_extraction_service import TopicExtractionService


class TestTopicExtractionService:
    """Test cases for TopicExtractionService."""

    @pytest.fixture
    def service(self):
        """Create a TopicExtractionService instance."""
        return TopicExtractionService()

    def test_init(self, service):
        """Test service initialization."""
        assert service.intermediate_prompt_path.endswith("intermediate-topics-prompt.txt")
        assert service.final_prompt_path.endswith("final-topics-prompt-onepass.txt")

    @patch('webinar_processor.services.topic_extraction_service.create_summary')
    @patch('webinar_processor.services.topic_extraction_service.create_summary_with_context')
    @patch('webinar_processor.services.topic_extraction_service.LLMConfig.get_model')
    def test_extract_topics_two_stage_process(
        self,
        mock_llm_config_get_model,
        mock_create_summary_with_context,
        mock_create_summary,
        service
    ):
        """Test extract_topics calls both stages correctly."""
        mock_llm_config_get_model.return_value = 'test-model'
        mock_create_summary.return_value = 'intermediate result'
        mock_create_summary_with_context.return_value = 'final result'

        result = service.extract_topics('test text', language='ru')

        # Verify both stages were called
        mock_create_summary.assert_called_once()
        mock_create_summary_with_context.assert_called_once()

        # Verify result
        assert result == ('intermediate result', 'final result')

    @patch('webinar_processor.services.topic_extraction_service.create_summary')
    @patch('webinar_processor.services.topic_extraction_service.create_summary_with_context')
    @patch('webinar_processor.services.topic_extraction_service.LLMConfig.get_model')
    def test_extract_topics_with_model_override(
        self,
        mock_llm_config_get_model,
        mock_create_summary_with_context,
        mock_create_summary,
        service
    ):
        """Test extract_topics respects model override."""
        service.extract_topics('test text', model='custom-model')

        # Should not call config when model is provided
        mock_llm_config_get_model.assert_not_called()
