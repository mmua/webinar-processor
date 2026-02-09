import openai
from typing import Optional
import logging

from .config import LLMConfig
from .exceptions import LLMError, TokenLimitError
from .constants import TOKEN_LIMITS
from ..utils.token import count_tokens

logger = logging.getLogger(__name__)

class LLMClient:
    _NONE_RESPONSES = frozenset({'none', 'null', 'n/a', ''})

    def __init__(self):
        LLMConfig.validate()
        self.client = openai.OpenAI(
            api_key=LLMConfig.get_api_key(),
            base_url=LLMConfig.get_base_url()
        )

    def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 100) -> Optional[str]:
        if model is None:
            model = LLMConfig.get_model('default')

        # Check token limit before making API call (prompt + completion must fit)
        token_limit = TOKEN_LIMITS.get(model, 128000)
        prompt_tokens = count_tokens(model, prompt)
        if prompt_tokens + max_tokens > token_limit:
            raise TokenLimitError(
                f"Prompt exceeds token limit: {prompt_tokens} + {max_tokens} = {prompt_tokens + max_tokens} > {token_limit} for model {model}"
            )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            logger.debug(f"LLM response for model {model}: {content[:100] if content else 'None'}...")
            if content is None:
                logger.warning(f"LLM returned None content for model {model}")
                return None
            return content.strip()
        except Exception as e:
            logger.error(f"LLM error for model {model}: {e}")
            raise LLMError(f"LLM generation failed for model {model}: {e}") from e
    
    def extract_speaker_name(self, text: str) -> Optional[str]:
        prompt = f'Analyze the following text and extract the speaker\'s name if they introduce themselves. Look for patterns like "Hi, I\'m [name]", "My name is [name]", "This is [name]", "[name] here", "I\'m [name]". If no clear self-introduction is found, respond with "None". Only return the name itself, not titles or additional words. Return the name in the same language as the text.\n\nText: "{text}"\n\nName:'
        try:
            response = self.generate(prompt, model=LLMConfig.get_model('speaker_extraction'), max_tokens=50)
        except LLMError:
            logger.warning("Failed to extract speaker name due to LLM error")
            return None

        if response and response.lower() not in self._NONE_RESPONSES:
            return response.strip('"\'.,!?') or None
        return None
