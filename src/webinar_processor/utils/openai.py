import logging

from webinar_processor.llm import LLMClient, LLMConfig, OUTPUT_LIMITS
from webinar_processor.llm.exceptions import TokenLimitError
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

_llm_client = None

def get_output_limit(model: str) -> int:
    return OUTPUT_LIMITS.get(model, 4096)

def get_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

@retry(wait=wait_random_exponential(multiplier=1, min=30, max=120), stop=stop_after_attempt(7), retry=retry_if_not_exception_type(TokenLimitError))
def get_completion(prompt, model=None, max_tokens=None):
    """
    Generate a completion using the LLM client.

    Args:
        prompt: The text prompt to send to the LLM
        model: The model name to use (defaults to 'default' from config)
        max_tokens: Maximum tokens in response (defaults to model's output limit)

    Returns:
        The generated text response from the LLM

    Raises:
        LLMError: When LLM generation fails (e.g., API errors, network issues)
    """
    client = get_client()
    if model is None:
        model = LLMConfig.get_model('default')
    if max_tokens is None:
        max_tokens = get_output_limit(model)
    return client.generate(prompt, model=model, max_tokens=max_tokens)
