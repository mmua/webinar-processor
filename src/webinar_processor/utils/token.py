"""Token utilities for LLM operations."""

import tiktoken
from typing import Union


def count_tokens(model: str, text: str) -> int:
    """
    Count tokens in text using tiktoken, with fallback for unknown models.

    Args:
        model: The model name to use for tokenization
        text: The text to tokenize

    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))
