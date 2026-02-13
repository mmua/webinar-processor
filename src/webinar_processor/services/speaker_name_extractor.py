"""Extract speaker names from transcript text using LLM."""

import logging
from typing import Optional

from webinar_processor.llm.config import LLMConfig
from webinar_processor.llm.exceptions import LLMError

logger = logging.getLogger(__name__)

_NONE_RESPONSES = frozenset({'none', 'null', 'n/a', ''})


def extract_speaker_name(text: str, client=None) -> Optional[str]:
    """
    Analyze text and extract the speaker's name if they introduce themselves.

    Looks for patterns like "Hi, I'm [name]", "My name is [name]", etc.
    Returns the name in the same language as the text.

    Args:
        text: Text to search for self-introductions
        client: Optional LLMClient instance. Uses singleton if not provided.

    Returns:
        Extracted name, or None if no introduction found
    """
    if client is None:
        from webinar_processor.utils.completion import get_client
        client = get_client()

    prompt = (
        'Analyze the following text and extract the speaker\'s name if they '
        'introduce themselves. Look for patterns like "Hi, I\'m [name]", '
        '"My name is [name]", "This is [name]", "[name] here", "I\'m [name]". '
        'If no clear self-introduction is found, respond with "None". '
        'Only return the name itself, not titles or additional words. '
        'Return the name in the same language as the text.\n\n'
        f'Text: "{text}"\n\nName:'
    )

    try:
        response = client.generate(
            prompt,
            model=LLMConfig.get_model('speaker_extraction'),
            max_tokens=50,
        )
    except LLMError:
        logger.warning("Failed to extract speaker name due to LLM error")
        return None

    if response and response.lower() not in _NONE_RESPONSES:
        return response.strip('"\'.,!?') or None
    return None
