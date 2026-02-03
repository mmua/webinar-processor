import openai
from typing import Optional
import logging

from .config import LLMConfig

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
            # Make error visible in console, not just logger
            import click
            click.echo(click.style(f"LLM EXCEPTION for {model}: {type(e).__name__}: {e}", fg='red'), err=True)
            logger.error(f"LLM error for model {model}: {e}")
            return None
    
    def extract_speaker_name(self, text: str) -> Optional[str]:
        prompt = f'Analyze the following text and extract the speaker\'s name if they introduce themselves. Look for patterns like "Hi, I\'m [name]", "My name is [name]", "This is [name]", "[name] here", "I\'m [name]". If no clear self-introduction is found, respond with "None". Only return the name itself, not titles or additional words. Return the name in the same language as the text.\n\nText: "{text}"\n\nName:'
        response = self.generate(prompt, model=LLMConfig.get_model('speaker_extraction'), max_tokens=50)
        
        if response and response.lower() not in self._NONE_RESPONSES:
            return response.strip('"\'.,!?') or None
        return None
