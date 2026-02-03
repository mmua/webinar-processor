"""LLM context exceptions."""


class LLMError(Exception):
    """Raised when LLM generation fails."""
    pass


class TokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    pass
