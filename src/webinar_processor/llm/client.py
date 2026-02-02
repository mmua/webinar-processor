import openai
from typing import Optional
import logging

from .config import LLMConfig

logger = logging.getLogger(__name__)

class LLMClient:
    _NONE_RESPONSES = frozenset({'none', 'null', 'n/a', ''})
    
    def __init__(self):
        LLMConfig.validate()
        self.client = openai.OpenAI(api_key=LLMConfig.api_key, base_url=LLMConfig.base_url)
    
    def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 100, temperature: float = 0.3) -> Optional[str]:
        if model is None:
            model = LLMConfig.get_model('default')
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None
    
    def extract_speaker_name(self, text: str) -> Optional[str]:
        prompt = f'Analyze the following text and extract the speaker\'s name if they introduce themselves. Look for patterns like "Hi, I\'m [name]", "My name is [name]", "This is [name]", "[name] here", "I\'m [name]". If no clear self-introduction is found, respond with "None". Only return the name itself, not titles or additional words. Return the name in the same language as the text.\n\nText: "{text}"\n\nName:'
        response = self.generate(prompt, model=LLMConfig.get_model('speaker_extraction'), max_tokens=50, temperature=0.1)
        
        if response and response.lower() not in self._NONE_RESPONSES:
            return response.strip('"\'.,!?') or None
        return None
