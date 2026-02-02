import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfig:
    api_key = os.getenv('LLM_API_KEY')
    base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
    
    @staticmethod
    def _get_env_or_default(var_name: str, default: str) -> str:
        """Get environment variable value, treating empty strings as unset."""
        value = os.getenv(var_name)
        return value if value else default
    
    models = {
        'summarization': _get_env_or_default('LLM_SUMMARIZATION_MODEL', 'gpt-5.2-mini'),
        'topics': _get_env_or_default('LLM_TOPICS_MODEL', 'gpt-5.2'),
        'quiz': _get_env_or_default('LLM_QUIZ_MODEL', 'gpt-5.2'),
        'story': _get_env_or_default('LLM_STORY_MODEL', 'gpt-5.2'),
        'speaker_extraction': _get_env_or_default('LLM_SPEAKER_EXTRACTION_MODEL', 'gpt-5.2-mini'),
    }
    
    @classmethod
    def get_model(cls, task: str) -> str:
        return cls.models.get(task, cls._get_env_or_default('LLM_DEFAULT_MODEL', 'gpt-5.2-mini'))
    
    @classmethod
    def validate(cls):
        if not cls.api_key:
            raise ValueError('LLM_API_KEY environment variable is required')
