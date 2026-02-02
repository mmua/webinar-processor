import os
import openai
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM-based text analysis and generation."""
    
    _NONE_RESPONSES = frozenset({'none', 'null', 'n/a', ''})

    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Initialize the LLM service.
        
        Args:
            model: The model to use for generation (default: gpt-4.1-mini)
        """
        self.model = model
        self.client = None
        
        # Try to initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            logger.warning("OPENAI_API_KEY not found. LLM features will be disabled.")
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3) -> Optional[str]:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Generated text or None if service unavailable
        """
        if not self.client:
            logger.warning("LLM service not available")
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except openai.APIError as e:
            logger.error(f"OpenAI API Error generating LLM response: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating LLM response: {str(e)}")
            return None
    
    def extract_speaker_name(self, text: str) -> Optional[str]:
        """
        Extract speaker name from self-introduction text.
        
        Args:
            text: Speaker's text content
            
        Returns:
            Extracted name or None if no introduction found
        """
        prompt = f"""Analyze the following text and extract the speaker's name if they introduce themselves.
Look for patterns like:
- "Hi, I'm [name]"
- "My name is [name]"
- "This is [name]"
- "[name] here"
- "I'm [name]"

If no clear self-introduction is found, respond with "None".
Only return the name itself, not titles or additional words.
Return the name in the same language as the text.

Text: "{text}"

Name:"""
        
        response = self.generate(prompt, max_tokens=50, temperature=0.1)
        
        if response and response.lower() not in self._NONE_RESPONSES:
            # Clean up the response
            name = response.strip('"\'.,!?')
            return name if name else None
        
        return None 