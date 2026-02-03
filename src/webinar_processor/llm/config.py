import os
from dotenv import load_dotenv

load_dotenv()

# Default models for each task type
_DEFAULT_MODELS = {
    'summarization': 'gpt-5-mini',
    'topics': 'gpt-5.2',
    'quiz': 'gpt-5.2',
    'story': 'gpt-5.2',
    'speaker_extraction': 'gpt-5-mini',
    'default': 'gpt-5-mini',
}


class LLMConfig:
    @classmethod
    def get_api_key(cls) -> str | None:
        """Get API key, supporting both new and legacy environment variables."""
        return os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')

    @classmethod
    def get_base_url(cls) -> str:
        """Get base URL for the LLM API."""
        return os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')

    @classmethod
    def get_model(cls, task: str) -> str:
        """Get model for a task, checking environment variables dynamically."""
        # Check task-specific env var first
        env_var = f'LLM_{task.upper()}_MODEL'
        model = os.getenv(env_var)
        if model:
            return model

        # Fall back to default model env var
        default_model = os.getenv('LLM_DEFAULT_MODEL')
        if default_model:
            return default_model

        # Fall back to hardcoded defaults
        return _DEFAULT_MODELS.get(task, _DEFAULT_MODELS['default'])

    @classmethod
    def validate(cls):
        if not cls.get_api_key():
            raise ValueError(
                'LLM API key not found. Set LLM_API_KEY or OPENAI_API_KEY environment variable.'
            )
