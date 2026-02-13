"""Shared transcript loading and formatting logic.

Used by summarize, storytell, and quiz commands to avoid
duplicating the load-JSON / detect-format / format pattern.
"""

import json
import logging
from typing import Optional, Tuple

from webinar_processor.utils.transcript_formatter import (
    is_diarized_format,
    format_diarized_transcript,
    add_paragraph_breaks,
)

logger = logging.getLogger(__name__)


def load_and_format_transcript(
    asr_file: str,
) -> Tuple[str, Optional[list]]:
    """
    Load a transcript file and format it into text for LLM processing.

    Handles both input formats:
    - Diarized: JSON array with start/end/speaker/text segments
    - ASR: JSON object with "text" field

    Args:
        asr_file: Path to JSON transcript file

    Returns:
        Tuple of (formatted_text, segments_or_none).
        segments is the original list for diarized input, None for ASR.

    Raises:
        FileNotFoundError: If transcript file does not exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If transcript contains no processable text
    """
    with open(asr_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if is_diarized_format(data):
        logger.info("Diarized transcript detected, formatting...")
        text = format_diarized_transcript(data)
        segments = data
    else:
        text = data.get("text", "")
        segments = None
        if '\n' not in text and len(text) > 1000:
            logger.info("Flat text, adding paragraph breaks...")
            text = add_paragraph_breaks(text)

    if not text.strip():
        raise ValueError("No text to process in transcript")

    return text, segments
