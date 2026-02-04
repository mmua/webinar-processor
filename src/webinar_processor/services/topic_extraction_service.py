"""Service for extracting topics from transcripts using LLM."""

from typing import Optional, Tuple
from webinar_processor.llm import LLMConfig
from webinar_processor.utils.openai import create_summary, create_summary_with_context
from webinar_processor.utils.package import get_config_path


class TopicExtractionService:
    """Service for extracting topics from transcripts using LLM."""

    def __init__(self):
        self.intermediate_prompt_path = get_config_path("intermediate-topics-prompt.txt")
        self.final_prompt_path = get_config_path("final-topics-prompt-onepass.txt")

    def extract_topics(
        self,
        text: str,
        language: str = "ru",
        model: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Extract topics from text in two stages.

        Args:
            text: Transcript text to analyze
            language: Language code (default: "ru")
            model: LLM model name (defaults to config)

        Returns:
            Tuple of (intermediate_topics, final_topics)

        Raises:
            IOError: If prompt files cannot be read
            webinar_processor.llm.LLMError: If LLM calls fail
        """
        model = model or LLMConfig.get_model('topics')

        # Stage 1: Intermediate extraction
        with open(self.intermediate_prompt_path, "r", encoding="utf-8") as pf:
            intermediate_prompt = pf.read()

        intermediate_topics = create_summary(text, language, model, intermediate_prompt)

        # Stage 2: Final refinement
        with open(self.final_prompt_path, "r", encoding="utf-8") as pf:
            final_prompt = pf.read()

        final_topics = create_summary_with_context(
            text, intermediate_topics, language, model, final_prompt
        )

        return intermediate_topics, final_topics
