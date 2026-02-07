"""LLM context constants."""

TOKEN_LIMITS = {
    'gpt-4.1': 1047576,
    'gpt-4.1-mini': 1047576,
    'gpt-4.1-nano': 1047576,
    'gpt-4o': 128000,
    'gpt-4o-mini': 128000,
    'gpt-4-turbo': 128000,
    'gpt-5.2': 256000,
    'gpt-5-mini': 128000,
    'qwen3-max-2026-01-23': 128000,
}

OUTPUT_LIMITS = {
    'gpt-4.1': 32768,
    'gpt-4.1-mini': 32768,
    'gpt-4.1-nano': 32768,
    'gpt-4o': 32768,
    'gpt-4o-mini': 16384,
    'gpt-4-turbo': 32768,
    'gpt-5.2': 128000,
    'gpt-5-mini': 64000,
    'qwen3-max-2026-01-23': 32000,
}
